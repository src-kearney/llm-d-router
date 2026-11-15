/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package approximateprefix

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"strconv"
	"strings"
	"testing"

	"github.com/alicebob/miniredis/v2"
	miniredisserver "github.com/alicebob/miniredis/v2/server"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"

	runtimedl "github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	redissync "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/cross_plugin/redis"
)

type replicationBench struct {
	ctx       context.Context
	producers []*dataProducer
	syncers   []*redissync.RedisSyncer
	windows   [][][]uint64
	bases     []uint64
	endpoints []fwksched.Endpoint
	endpoint  fwksched.Endpoint
	request   *fwksched.InferenceRequest
	rng       *rand.Rand
	reads     int
}

type replicationBenchOptions struct {
	peers        int
	endpoints    int
	overlap      int
	queryBlocks  int
	matchBlocks  int
	redisAddress string
}

// Detect sequence support to benchmark both transport revisions with the same source.
func replicationBenchModes() []string {
	if _, ok := any(&attrprefix.PrefixHashSet{}).(interface{ SequenceBase() uint64 }); ok {
		return []string{"full", "incremental"}
	}
	return []string{"full"}
}

func newReplicationBench(t testing.TB, mode string, prompts, hashes int) *replicationBench {
	return newReplicationBenchWithOptions(t, mode, prompts, hashes, replicationBenchOptions{peers: 1})
}

func newReplicationBenchWithOptions(t testing.TB, mode string, prompts, hashes int, options replicationBenchOptions) *replicationBench {
	t.Helper()
	if options.endpoints == 0 {
		options.endpoints = 1
	}
	if options.queryBlocks == 0 {
		options.queryBlocks = 4
	}
	ctx := logr.NewContext(t.Context(), logr.Discard())
	redis := miniredis.RunT(t)
	// miniredis supports HEXPIRE; the Redis package's client compatibility hook
	// is private, so factory-created clients use the same conversion at the server.
	redis.Server().SetPreHook(func(peer *miniredisserver.Peer, command string, args ...string) bool {
		if command != "HPEXPIRE" {
			return false
		}
		millis, err := strconv.ParseInt(args[1], 10, 64)
		if err != nil {
			peer.WriteError(err.Error())
			return true
		}
		rewritten := append([]string{"HEXPIRE"}, args...)
		rewritten[2] = strconv.FormatInt((millis+999)/1000, 10)
		redis.Server().Dispatch(peer, rewritten)
		return true
	})
	_, endpoints := benchPods(options.endpoints)
	address := options.redisAddress
	if address == "" {
		address = redis.Addr()
	}
	f := &replicationBench{
		ctx:       ctx,
		producers: make([]*dataProducer, options.peers+1),
		syncers:   make([]*redissync.RedisSyncer, options.peers+1),
		windows:   make([][][]uint64, options.peers+1),
		bases:     make([]uint64, options.peers+1),
		endpoints: endpoints,
		endpoint:  endpoints[0],
		rng:       rand.New(rand.NewPCG(1, 2)),
		request: &fwksched.InferenceRequest{
			RequestID:   "bench",
			TargetModel: "bench-model",
			Body:        tokenizedBody(benchTokens(1, options.queryBlocks*benchBlockSize)),
		},
	}
	for replica := range f.producers {
		cfg := defaultConfig
		cfg.AutoTune = false
		cfg.MaxSyncedTokens = prompts * hashes * cfg.BlockSizeTokens
		p, err := newDataProducer(ctx, ApproxPrefixCachePluginType, cfg, testHandleWithSyncer())
		require.NoError(t, err)
		f.producers[replica] = p
		params := map[string]any{"address": address}
		if mode == "incremental" {
			params["windowReplication"] = mode
		}
		if options.redisAddress != "" {
			// Race-instrumented fixture checks can exceed the default state TTL.
			params["stateTTL"] = "1m"
		}
		encoded, err := json.Marshal(params)
		require.NoError(t, err)
		instance, err := redissync.RedisSyncerFactory("bench", json.NewDecoder(strings.NewReader(string(encoded))), plugin.NewEppHandle(ctx, nil))
		require.NoError(t, err)
		f.syncers[replica] = instance.(*redissync.RedisSyncer)
		for i := range prompts {
			prompt := f.prompt(hashes)
			if replica > 0 && i < prompts*options.overlap/100 {
				prompt = f.windows[0][i]
			}
			if replica == 0 && i == prompts-1 && options.matchBlocks > 0 {
				query := benchHashes(p, benchTokens(1, options.queryBlocks*benchBlockSize))
				require.LessOrEqual(t, options.matchBlocks, min(len(query), len(prompt)))
				for j := range options.matchBlocks {
					prompt[j] = uint64(query[j])
				}
			}
			f.append(replica, prompt, prompts)
		}
	}
	p := f.producers[0]
	for _, endpoint := range f.endpoints {
		endpoint.Put(p.syncHashDK, &fwkdl.DynamicAttribute{Get: func() fwkdl.Cloneable {
			f.reads++
			value, ok, err := f.syncers[0].Get(ctx, p.CrossReplicaState(), endpoint.GetMetadata().ID.String())
			if err != nil || !ok {
				t.Fatalf("Get: found=%t err=%v", ok, err)
			}
			return value.(fwkdl.Cloneable)
		}})
	}
	runtimedl.RegisterScopeSpecs([]plugin.Plugin{p})
	f.refresh(t)
	streams := 0
	for _, key := range redis.Keys() {
		if redis.Type(key) == "stream" {
			streams++
		}
	}
	if options.redisAddress == "" && mode == "incremental" {
		require.Equal(t, len(f.producers)*len(f.endpoints), streams, "each replica must publish one stream per endpoint")
	} else {
		require.Zero(t, streams, "full replication must not create streams")
	}
	require.NoError(t, f.produce())
	if options.matchBlocks > 0 {
		// Peer ingestion can evict local entries; keep the intended hit resident.
		for _, endpoint := range f.endpoints {
			p.indexerInst.Add(uint64sToBlockHashes(f.windows[0][prompts-1]), p.makeserver(endpoint))
		}
		require.NoError(t, f.produce())
	}
	f.check(t)
	for _, endpoint := range f.endpoints {
		value, ok := endpoint.Get(p.dk)
		require.True(t, ok)
		require.Equal(t, options.matchBlocks, value.(*attrprefix.PrefixCacheMatchInfo).MatchBlocks())
	}
	return f
}

func (f *replicationBench) id() string { return f.endpoint.GetMetadata().ID.String() }

func (f *replicationBench) prompt(size int) []uint64 {
	out := make([]uint64, size)
	for i := range out {
		out[i] = f.rng.Uint64()
	}
	return out
}

func (f *replicationBench) append(replica int, prompt []uint64, capacity int) {
	hashes := uint64sToBlockHashes(prompt)
	p := f.producers[replica]
	for _, endpoint := range f.endpoints {
		p.indexerInst.Add(hashes, p.makeserver(endpoint))
		p.recent.record(endpoint.GetMetadata().ID.String(), hashes)
	}
	f.windows[replica] = append(f.windows[replica], prompt)
	if len(f.windows[replica]) > capacity {
		f.windows[replica] = f.windows[replica][1:]
		f.bases[replica]++
	}
}

func (f *replicationBench) refresh(t testing.TB) {
	t.Helper()
	for i := range f.producers {
		replica := (i + 1) % len(f.producers)
		for _, endpoint := range f.endpoints {
			require.NoError(t, f.syncers[replica].Set(f.ctx, f.producers[replica].CrossReplicaState(), endpoint.GetMetadata().ID.String()))
		}
	}
}

func (f *replicationBench) produce() error {
	p := f.producers[0]
	request, endpoints, violations := runtimedl.ScopeInvocation(logr.Discard(), "Produce", p, f.request, f.endpoints)
	if err := p.Produce(f.ctx, request, endpoints); err != nil {
		return err
	}
	return violations.Write()
}

func (f *replicationBench) expected() [][]uint64 {
	var out [][]uint64
	seen := make(map[string]bool)
	for _, window := range f.windows {
		for _, prompt := range window {
			key := fmt.Sprint(prompt)
			if !seen[key] {
				seen[key] = true
				out = append(out, prompt)
			}
		}
	}
	return out
}

func (f *replicationBench) check(t testing.TB) {
	t.Helper()
	p := f.producers[0]
	_, endpoints, _ := runtimedl.ScopeInvocation(logr.Discard(), "Produce", p, f.request, f.endpoints)
	for _, endpoint := range endpoints {
		value, ok := endpoint.Get(p.syncHashDK)
		require.True(t, ok)
		set := value.(*attrprefix.PrefixHashSet)
		require.ElementsMatch(t, f.expected(), set.Prompts)
		require.True(t, p.peerMergedDigestMatches(endpoint.GetMetadata().ID.String(), set.ContentDigest()))
		for i, p := range f.producers {
			local := p.CrossReplicaState().Read(endpoint.GetMetadata().ID.String()).(*attrprefix.PrefixHashSet)
			require.Equal(t, f.windows[i], local.Prompts)
			if sequence, ok := any(local).(interface {
				SequenceBase() uint64
				SequenceTip() uint64
			}); ok {
				require.Equal(t, f.bases[i], sequence.SequenceBase())
				require.Equal(t, f.bases[i]+uint64(len(f.windows[i])), sequence.SequenceTip())
			}
		}
	}
}

func TestReplicationBenchSequencesAndScores(t *testing.T) {
	for _, mode := range replicationBenchModes() {
		t.Run(mode, func(t *testing.T) {
			f := newReplicationBench(t, mode, 4, 4)
			for replica := range f.producers {
				tokens := benchTokens(uint32(replica+10), 4*benchBlockSize)
				f.request.Body = tokenizedBody(tokens)
				require.NoError(t, f.produce())
				value, ok := f.endpoint.Get(f.producers[0].dk)
				require.True(t, ok)
				require.Zero(t, value.(*attrprefix.PrefixCacheMatchInfo).MatchBlocks())
				hashes := benchHashes(f.producers[replica], tokens)
				prompt := promptsToUint64([][]blockHash{hashes})[0]
				f.append(replica, prompt, 4)
				if replica == 1 {
					f.refresh(t)
				}
				for range 2 {
					before := f.reads
					require.NoError(t, f.produce())
					require.Equal(t, before+1, f.reads)
					f.check(t)
					value, ok = f.endpoint.Get(f.producers[0].dk)
					require.True(t, ok)
					require.Equal(t, 4, value.(*attrprefix.PrefixCacheMatchInfo).MatchBlocks())
				}
			}
		})
	}
}

// BenchmarkReplicationLocalWork excludes fixture construction and initial publication.
// Produce includes scope creation and reuses a fixed request and endpoint.
// Changed cases exclude recording, peer refresh, and validation; their cache and
// GC effects can still affect the timed request. Read and Aggregate use fixed inputs.
func BenchmarkReplicationLocalWork(b *testing.B) {
	for _, shape := range []struct{ prompts, hashes int }{{125, 250}, {1000, 500}} {
		for _, mode := range replicationBenchModes() {
			b.Run(fmt.Sprintf("hashes=%d/mode=%s", shape.prompts*shape.hashes, mode), func(b *testing.B) {
				for _, operation := range []string{"Read", "Aggregate", "ProduceUnchanged", "ProduceLocalChanged", "ProducePeerChanged"} {
					b.Run(operation, func(b *testing.B) {
						f := newReplicationBench(b, mode, shape.prompts, shape.hashes)
						spec := f.producers[0].CrossReplicaState()
						values := []any{spec.Read(f.id()), f.producers[1].CrossReplicaState().Read(f.id())}
						var result any
						b.ReportAllocs()
						switch operation {
						case "Read":
							for b.Loop() {
								result = spec.Read(f.id())
							}
							require.Equal(b, f.windows[0], result.(*attrprefix.PrefixHashSet).Prompts)
						case "Aggregate":
							for b.Loop() {
								result = spec.Aggregate(values)
							}
							require.Equal(b, f.expected(), result.(*attrprefix.PrefixHashSet).Prompts)
						case "ProduceUnchanged":
							for b.Loop() {
								if err := f.produce(); err != nil {
									b.Fatal(err)
								}
							}
							f.check(b)
						case "ProduceLocalChanged", "ProducePeerChanged":
							replica := 0
							if operation == "ProducePeerChanged" {
								replica = 1
							}
							for b.Loop() {
								b.StopTimer()
								f.append(replica, f.prompt(shape.hashes), shape.prompts)
								if replica == 1 {
									f.refresh(b)
								}
								digest := (&attrprefix.PrefixHashSet{Prompts: f.expected()}).ContentDigest()
								require.False(b, f.producers[0].peerMergedDigestMatches(f.id(), digest))
								b.StartTimer()
								if err := f.produce(); err != nil {
									b.Fatal(err)
								}
								b.StopTimer()
								f.check(b)
								b.StartTimer()
							}
						}
					})
				}
			})
		}
	}
}
