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
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/go-logr/logr"
	goredis "github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"

	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// Each case changes one dimension from a 125-prompt, 250-hash baseline.
// Prompt-count cases hold total hashes fixed; hash-count cases hold prompt length fixed.
func BenchmarkReplicationShape(b *testing.B) {
	cases := []struct {
		name            string
		prompts, hashes int
		options         replicationBenchOptions
	}{
		{"baseline", 125, 250, replicationBenchOptions{peers: 1}},
		{"prompts=25", 25, 1250, replicationBenchOptions{peers: 1}},
		{"prompts=625", 625, 50, replicationBenchOptions{peers: 1}},
		{"hashes=125000", 500, 250, replicationBenchOptions{peers: 1}},
		{"hashes=500000", 2000, 250, replicationBenchOptions{peers: 1}},
		{"overlap=50", 125, 250, replicationBenchOptions{peers: 1, overlap: 50}},
		{"overlap=100", 125, 250, replicationBenchOptions{peers: 1, overlap: 100}},
		{"peers=3", 125, 250, replicationBenchOptions{peers: 3}},
		{"peers=7", 125, 250, replicationBenchOptions{peers: 7}},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			f := newReplicationBenchWithOptions(b, "full", tc.prompts, tc.hashes, tc.options)
			spec := f.producers[0].CrossReplicaState()
			values := make([]any, len(f.producers))
			for i, p := range f.producers {
				values[i] = p.CrossReplicaState().Read(f.id())
			}
			if tc.options.peers == 1 && tc.options.overlap == 0 {
				b.Run("Read", func(b *testing.B) {
					var result any
					b.ReportAllocs()
					for b.Loop() {
						result = spec.Read(f.id())
					}
					require.Equal(b, f.windows[0], result.(*attrprefix.PrefixHashSet).Prompts)
				})
			}
			b.Run("Aggregate", func(b *testing.B) {
				var result any
				b.ReportAllocs()
				for b.Loop() {
					result = spec.Aggregate(values)
				}
				require.Equal(b, f.expected(), result.(*attrprefix.PrefixHashSet).Prompts)
			})
		})
	}
}

// Produce includes scope creation and warmed dynamic Get/clone/digest work.
// Endpoint count is candidate count in one request, not independent request throughput.
func BenchmarkReplicationRequest(b *testing.B) {
	cases := []struct {
		name    string
		options replicationBenchOptions
	}{
		{"baseline", replicationBenchOptions{peers: 1}},
		{"endpoints=8", replicationBenchOptions{peers: 1, endpoints: 8}},
		{"endpoints=16", replicationBenchOptions{peers: 1, endpoints: 16}},
		{"peers=3", replicationBenchOptions{peers: 3}},
		{"peers=7", replicationBenchOptions{peers: 7}},
		{"overlap=50", replicationBenchOptions{peers: 1, overlap: 50}},
		{"overlap=100", replicationBenchOptions{peers: 1, overlap: 100}},
		{"query=128/match=0", replicationBenchOptions{peers: 1, queryBlocks: 128}},
		{"query=128/match=128", replicationBenchOptions{peers: 1, queryBlocks: 128, matchBlocks: 128}},
		{"query=512/match=0", replicationBenchOptions{peers: 1, queryBlocks: 512}},
		{"query=512/match=128", replicationBenchOptions{peers: 1, queryBlocks: 512, matchBlocks: 128}},
	}
	for _, mode := range replicationBenchModes() {
		for _, tc := range cases {
			b.Run("mode="+mode+"/"+tc.name, func(b *testing.B) {
				f := newReplicationBenchWithOptions(b, mode, 125, 250, tc.options)
				b.ReportAllocs()
				for b.Loop() {
					if err := f.produce(); err != nil {
						b.Fatal(err)
					}
				}
				f.check(b)
			})
		}
	}
}

// Each parallel operation performs Get, with one local append per 16 operations
// in mixed mode. This measures shared window-lock contention and includes writes.
// Peer cache is fixed; fixture bookkeeping and endpoint attributes are not shared.
func BenchmarkReplicationConcurrentGet(b *testing.B) {
	for _, mode := range replicationBenchModes() {
		for _, mixed := range []bool{false, true} {
			b.Run(fmt.Sprintf("mode=%s/mixed=%t", mode, mixed), func(b *testing.B) {
				f := newReplicationBench(b, mode, 125, 250)
				p := f.producers[0]
				spec := p.CrossReplicaState()
				id := f.id()
				var workers, writes atomic.Uint64
				b.ReportAllocs()
				b.ResetTimer()
				b.RunParallel(func(pb *testing.PB) {
					worker := workers.Add(1)
					prompt := make([]blockHash, 250)
					var iteration uint64
					for pb.Next() {
						iteration++
						if mixed && iteration%16 == 0 {
							for i := range prompt {
								prompt[i] = blockHash(worker<<48 | iteration<<16 | uint64(i))
							}
							p.recent.record(id, prompt)
							writes.Add(1)
						}
						value, ok, err := f.syncers[0].Get(f.ctx, spec, id)
						if err != nil || !ok || value == nil {
							b.Errorf("Get: found=%t err=%v", ok, err)
							return
						}
					}
				})
				b.ReportMetric(float64(writes.Load())/float64(b.N), "writes/op")
			})
		}
	}
}

// Readers retain snapshots while appends evict old generations. A prompt encodes
// its generation in every hash so torn prompts and noncontiguous windows are visible.
func TestReplicationConcurrentSnapshots(t *testing.T) {
	cfg := defaultConfig
	cfg.AutoTune = false
	cfg.MaxSyncedTokens = 16 * 8 * cfg.BlockSizeTokens
	p, err := newDataProducer(logr.NewContext(t.Context(), logr.Discard()), ApproxPrefixCachePluginType, cfg, testHandleWithSyncer())
	require.NoError(t, err)
	id := "concurrent-snapshot"
	record := func(generation uint64) {
		prompt := make([]blockHash, 8)
		for i := range prompt {
			prompt[i] = blockHash(generation + 1)
		}
		p.recent.record(id, prompt)
	}
	for i := range 16 {
		record(uint64(i))
	}
	spec := p.CrossReplicaState()
	start := make(chan struct{})
	midway := make(chan struct{})
	var readers, snapshotTaken sync.WaitGroup
	snapshotTaken.Add(4)
	snapshots := make(chan *attrprefix.PrefixHashSet, 4*128)
	for range 4 {
		readers.Go(func() {
			<-start
			snapshots <- spec.Read(id).(*attrprefix.PrefixHashSet)
			snapshotTaken.Done()
			<-midway
			for range 127 {
				snapshots <- spec.Read(id).(*attrprefix.PrefixHashSet)
			}
		})
	}
	close(start)
	snapshotTaken.Wait()
	for i := 16; i < 32; i++ {
		record(uint64(i))
	}
	close(midway)
	for i := 32; i < 2048; i++ {
		record(uint64(i))
	}
	readers.Wait()
	close(snapshots)
	for snapshot := range snapshots {
		require.Len(t, snapshot.Prompts, 16)
		first := snapshot.Prompts[0][0]
		for i, prompt := range snapshot.Prompts {
			require.Len(t, prompt, 8)
			for _, hash := range prompt {
				require.Equal(t, first+uint64(i), hash)
			}
		}
		if sequence, ok := any(snapshot).(interface {
			SequenceBase() uint64
			SequenceTip() uint64
		}); ok {
			require.Equal(t, first-1, sequence.SequenceBase())
			require.Equal(t, first-1+16, sequence.SequenceTip())
		}
	}
	final := spec.Read(id).(*attrprefix.PrefixHashSet)
	require.Equal(t, uint64(2048), final.Prompts[15][0])
}

// APPROX_PREFIX_BENCH_REDIS must name a disposable Redis 7.4+ server: each case
// clears its default database. One tick publishes both replicas and refreshes
// their peer caches. Initial publication, recording and checks are outside timing.
func BenchmarkReplicationRedisTick(b *testing.B) {
	address := os.Getenv("APPROX_PREFIX_BENCH_REDIS")
	if address == "" {
		b.Skip("set APPROX_PREFIX_BENCH_REDIS to an isolated Redis server")
	}
	for _, mode := range replicationBenchModes() {
		for _, shape := range []struct{ prompts, hashes int }{{125, 250}, {1000, 500}} {
			for _, change := range []string{"unchanged", "one-prompt"} {
				b.Run(fmt.Sprintf("mode=%s/hashes=%d/%s", mode, shape.prompts*shape.hashes, change), func(b *testing.B) {
					client := goredis.NewClient(&goredis.Options{Addr: address})
					b.Cleanup(func() { _ = client.Close() })
					require.NoError(b, client.FlushDB(b.Context()).Err())
					f := newReplicationBenchWithOptions(b, mode, shape.prompts, shape.hashes, replicationBenchOptions{peers: 1, redisAddress: address})
					// A second tick lets the replica that published first catch up.
					f.refresh(b)
					keys, err := client.Keys(b.Context(), "*").Result()
					require.NoError(b, err)
					streams := 0
					for _, key := range keys {
						kind, err := client.Type(b.Context(), key).Result()
						require.NoError(b, err)
						if kind == "stream" {
							streams++
						}
					}
					if mode == "incremental" {
						require.Equal(b, len(f.producers), streams)
					} else {
						require.Zero(b, streams)
					}
					b.ReportAllocs()
					for b.Loop() {
						b.StopTimer()
						if change == "one-prompt" {
							for replica := range f.producers {
								f.append(replica, f.prompt(shape.hashes), shape.prompts)
							}
						}
						b.StartTimer()
						f.refresh(b)
					}
					require.NoError(b, f.produce())
					f.check(b)
				})
			}
		}
	}
}

func TestReplicationSnapshotOwnership(t *testing.T) {
	f := newReplicationBench(t, "full", 2, 2)
	p := f.producers[0]
	spec := p.CrossReplicaState()
	id := f.id()
	p.recent.forget(id)
	input := []blockHash{11, 12}
	p.recent.record(id, input)
	first := spec.Read(id).(*attrprefix.PrefixHashSet)
	input[0] = 99
	require.Equal(t, [][]uint64{{11, 12}}, first.Prompts)
	require.Equal(t, first.Prompts, spec.Read(id).(*attrprefix.PrefixHashSet).Prompts)
	clone := first.Clone().(*attrprefix.PrefixHashSet)
	clone.Prompts[0][0] = 77
	clone.Prompts[0] = append(clone.Prompts[0], 78)
	require.Equal(t, [][]uint64{{11, 12}}, first.Prompts)
	p.recent.record(id, []blockHash{13, 14, 15, 16})
	require.Equal(t, [][]uint64{{13, 14, 15, 16}}, spec.Read(id).(*attrprefix.PrefixHashSet).Prompts)
	p.recent.forget(id)
	require.Empty(t, spec.Read(id).(*attrprefix.PrefixHashSet).Prompts)
	p.recent.record(id, []blockHash{21, 22})
	require.Equal(t, [][]uint64{{21, 22}}, spec.Read(id).(*attrprefix.PrefixHashSet).Prompts)
	require.Equal(t, [][]uint64{{11, 12}}, first.Prompts)
}

func TestReplicationAggregateOwnership(t *testing.T) {
	f := newReplicationBench(t, "full", 2, 2)
	spec := f.producers[0].CrossReplicaState()
	a := &attrprefix.PrefixHashSet{Prompts: [][]uint64{{1, 2}, {3}, {}, {1, 2}}}
	b := &attrprefix.PrefixHashSet{Prompts: [][]uint64{{1}, {2, 3}, {3}}}
	result := spec.Aggregate([]any{nil, a, b}).(*attrprefix.PrefixHashSet)
	require.Equal(t, [][]uint64{{1, 2}, {3}, {1}, {2, 3}}, result.Prompts)
	a.Prompts[0][0] = 99
	b.Prompts[0][0] = 98
	require.Equal(t, [][]uint64{{1, 2}, {3}, {1}, {2, 3}}, result.Prompts)
	result.Prompts[1][0] = 97
	require.Equal(t, uint64(3), a.Prompts[1][0])
}

func TestReplicationGetLifecycle(t *testing.T) {
	for _, mode := range replicationBenchModes() {
		t.Run(mode, func(t *testing.T) {
			f := newReplicationBench(t, mode, 2, 2)
			spec := f.producers[0].CrossReplicaState()
			get := func() *attrprefix.PrefixHashSet {
				value, ok, err := f.syncers[0].Get(f.ctx, spec, f.id())
				require.NoError(t, err)
				require.True(t, ok)
				return value.(*attrprefix.PrefixHashSet)
			}
			before := get().Clone().(*attrprefix.PrefixHashSet)
			require.ElementsMatch(t, f.expected(), before.Prompts)
			f.producers[0].recent.forget(f.id())
			f.producers[0].recent.record(f.id(), []blockHash{41, 42})
			f.windows[0] = [][]uint64{{41, 42}}
			require.ElementsMatch(t, f.expected(), get().Prompts)
			require.NoError(t, f.syncers[1].Delete(f.ctx, spec.StateKey, f.id()))
			require.NoError(t, f.syncers[0].Set(f.ctx, spec, f.id()))
			require.Equal(t, [][]uint64{{41, 42}}, get().Prompts)
			f.producers[1].recent.forget(f.id())
			f.producers[1].recent.record(f.id(), []blockHash{51, 52})
			f.windows[1] = [][]uint64{{51, 52}}
			f.refresh(t)
			require.ElementsMatch(t, f.expected(), get().Prompts)
			require.NotEqual(t, before.Prompts, get().Prompts)
		})
	}
}

func TestReplicationGetWindowBoundaries(t *testing.T) {
	for _, mode := range replicationBenchModes() {
		for _, peers := range []int{0, 1} {
			for _, empty := range []bool{false, true} {
				t.Run(fmt.Sprintf("mode=%s/peers=%d/empty-local=%t", mode, peers, empty), func(t *testing.T) {
					f := newReplicationBenchWithOptions(t, mode, 2, 2, replicationBenchOptions{peers: peers})
					require.Len(t, f.producers, peers+1)
					p := f.producers[0]
					spec := p.CrossReplicaState()
					if empty {
						p.recent.forget(f.id())
						f.windows[0] = nil
						f.bases[0] = 0
					}
					get := func() *attrprefix.PrefixHashSet {
						value, ok, err := f.syncers[0].Get(f.ctx, spec, f.id())
						require.NoError(t, err)
						require.True(t, ok)
						require.NotNil(t, value)
						return value.(*attrprefix.PrefixHashSet)
					}
					held := get()
					expected := held.Clone().(*attrprefix.PrefixHashSet)
					require.ElementsMatch(t, f.expected(), held.Prompts)
					public, ok := f.endpoint.Get(p.syncHashDK)
					require.True(t, ok)
					writable := public.(*attrprefix.PrefixHashSet)
					if len(writable.Prompts) > 0 {
						writable.Prompts[0][0] ^= 1
					}
					writable.Prompts = append(writable.Prompts, []uint64{99})
					require.ElementsMatch(t, f.expected(), get().Prompts)
					for i := range 3 {
						f.append(0, []uint64{uint64(100 + i), uint64(200 + i)}, 2)
					}
					require.ElementsMatch(t, f.expected(), get().Prompts)
					require.ElementsMatch(t, expected.Prompts, held.Prompts)
					require.NoError(t, f.produce())
					f.check(t)
				})
			}
		}
	}
}

// BenchmarkReplicationNoPeer measures direct Get and scoped Produce without a peer aggregate.
func BenchmarkReplicationNoPeer(b *testing.B) {
	for _, mode := range replicationBenchModes() {
		for _, operation := range []string{"Get", "Produce"} {
			b.Run("mode="+mode+"/"+operation, func(b *testing.B) {
				f := newReplicationBenchWithOptions(b, mode, 1000, 500, replicationBenchOptions{peers: 0})
				require.Len(b, f.producers, 1)
				spec := f.producers[0].CrossReplicaState()
				var result any
				b.ReportAllocs()
				for b.Loop() {
					if operation == "Get" {
						value, ok, err := f.syncers[0].Get(f.ctx, spec, f.id())
						if err != nil || !ok {
							b.Fatalf("Get: found=%t err=%v", ok, err)
						}
						result = value
					} else if err := f.produce(); err != nil {
						b.Fatal(err)
					}
				}
				if operation == "Get" {
					require.Equal(b, f.windows[0], result.(*attrprefix.PrefixHashSet).Prompts)
				}
				f.check(b)
			})
		}
	}
}
