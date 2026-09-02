# Tests for the LMDB-backed fallback store used by ObjCache on Julia without the runtime
# KV API. The store mirrors the runtime's objcache.cpp; the mechanics (layout, overwrite,
# eviction order, atime refresh) are exercised on temporary envs, process-level behaviour
# through child processes as in test/objcache.jl (which must be included first).

using CompilerCaching
using Test

const OC = CompilerCaching.ObjCache

@static if !OC.HAS_KV_API

const LF = OC.LMDBFallback
const MiB = 1024 * 1024

# A controllable clock: every call advances one second unless frozen.
mutable struct FakeClock
    t::Int64
    step::Int64
end
(c::FakeClock)() = (c.t += c.step; c.t)

@testset "ObjCache LMDB fallback" begin
    @testset "configuration" begin
        @test LF.configured_enabled(Dict{String,String}())
        @test LF.configured_enabled(Dict("JULIA_OBJCACHE" => "1"))
        @test !LF.configured_enabled(Dict("JULIA_OBJCACHE" => "0"))

        @test LF.configured_capacity(Dict{String,String}()) == LF.DEFAULT_CAPACITY == 512 * MiB
        @test LF.configured_capacity(Dict("JULIA_OBJCACHE_CAPACITY" => "4096")) == 4096
        @test LF.configured_capacity(Dict("JULIA_OBJCACHE_CAPACITY" => "0x1000")) == 4096
        @test (@test_logs (:warn, r"invalid value") LF.configured_capacity(
                   Dict("JULIA_OBJCACHE_CAPACITY" => "lots"))) == LF.DEFAULT_CAPACITY
        @test (@test_logs (:warn, r"invalid value") LF.configured_capacity(
                   Dict("JULIA_OBJCACHE_CAPACITY" => "0"))) == LF.DEFAULT_CAPACITY

        mktempdir() do dir
            @test LF.configured_path(Dict("JULIA_OBJCACHE_PATH" => dir)) ==
                  joinpath(abspath(dir), "fallback-lmdb1")
            default_path = LF.configured_path(Dict{String,String}())
            @test startswith(default_path, first(DEPOT_PATH))
            @test basename(default_path) == "v$(VERSION.major).$(VERSION.minor)"
        end
    end

    @testset "key framing matches the runtime" begin
        h = LF.entry_hash("ns", UInt8[1, 2])
        @test length(h) == 20
        @test h != LF.entry_hash("ns2", UInt8[1, 2])
        @test LF.entry_hash("ab", UInt8['c']) != LF.entry_hash("a", UInt8['b', 'c'])
        # SHA1("JLKV\0" ‖ u64be(2) ‖ "ns" ‖ [1,2]) computed independently
        @test bytes2hex(h) == bytes2hex(CompilerCaching.ObjCache.LMDBFallback.sha1(
            vcat(b"JLKV\0", UInt8[0, 0, 0, 0, 0, 0, 0, 2], b"ns", UInt8[1, 2])))

        ok = LF.objkey(h)
        @test length(ok) == LF.OBJKEY_SIZE && ok[1:2] == UInt8['O', 0] && ok[3:end] == h
        mk = LF.metakey(Int64(0x0102030405060708), h)
        @test length(mk) == LF.METAKEY_SIZE && mk[1:2] == UInt8['M', 0]
        @test mk[3:10] == UInt8[1, 2, 3, 4, 5, 6, 7, 8]          # big-endian time
        @test LF.metakey_parts(mk) == (Int64(0x0102030405060708), h)
        @test LF.is_metakey(mk) && !LF.is_metakey(ok)
        # Hit-bit times sort after every plain time.
        @test LF.metakey(Int64(1) | LF.HIT_BIT, h) > LF.metakey(typemax(Int32) |> Int64, h)
    end

    @testset "open / close / overwrite" begin
        mktempdir() do dir
            cache = LF.open(dir; capacity=4 * MiB)
            try
                @test LF.get(cache, "ns", b"hello") === nothing
                @test LF.put!(cache, "ns", b"hello", b"world")
                @test LF.get(cache, "ns", b"hello") == b"world"
                # Writes overwrite, as the runtime's mdb_put with flags 0 does.
                @test LF.put!(cache, "ns", b"hello", b"other")
                @test LF.get(cache, "ns", b"hello") == b"other"
                @test LF.get(cache, "other-ns", b"hello") === nothing
                @test LF.entries(cache) == 1
            finally
                LF.close(cache)
            end
            # Persistent across opens; the schema record matches.
            cache = LF.open(dir; capacity=4 * MiB)
            try
                @test LF.get(cache, "ns", b"hello") == b"other"
            finally
                LF.close(cache)
            end
        end
    end

    @testset "schema mismatch disables instead of wiping" begin
        mktempdir() do dir
            cache = LF.open(dir; capacity=4 * MiB)
            LF.put!(cache, "ns", b"k", b"v")
            LMDB = LF.LMDB
            LMDB.Transaction(cache.env) do txn
                LMDB.put!(txn, cache.objmeta, "schema", Int32(999))
            end
            LF.close(cache)
            @test_throws LF.SchemaMismatch LF.open(dir; capacity=4 * MiB)
            store = LF.LMDBStore()
            withenv("JULIA_OBJCACHE_PATH" => dir) do
                # configured_path appends a subdirectory; move the env there.
                sub = LF.configured_path()
                mkpath(sub)
                for f in ("data.mdb", "lock.mdb")
                    mv(joinpath(dir, f), joinpath(sub, f))
                end
                @test !OC.store_enabled(store)
                @test isfile(joinpath(sub, "data.mdb"))   # nothing was wiped
            end
        end
    end

    @testset "oversized values are dropped" begin
        mktempdir() do dir
            cache = LF.open(dir; capacity=256 * MiB)
            try
                big = zeros(UInt8, LF.EVICT_MAX_ENTRY_PAGES * cache.psize)
                @test LF.put!(cache, "ns", b"big", big)   # reported as accepted, like the runtime
                @test LF.get(cache, "ns", b"big") === nothing
                @test LF.entries(cache) == 0
            finally
                LF.close(cache)
            end
        end
    end

    @testset "eviction: never-read entries go first, then least recently used" begin
        mktempdir() do dir
            clock = FakeClock(1_000_000, 1)
            capacity = 4 * MiB
            cache = LF.open(dir; capacity, clock)
            try
                blob = 50_000
                # 40 entries (~2 MiB): read the second half so they carry the hit bit.
                for i in 1:40
                    LF.put!(cache, "ns", UInt8[1, i], rand(UInt8, blob))
                end
                for i in 21:40
                    @test LF.get(cache, "ns", UInt8[1, i]) !== nothing
                end
                # Then write enough to force eviction (another ~4 MiB).
                for i in 1:80
                    LF.put!(cache, "ns", UInt8[2, i], rand(UInt8, blob))
                end
                @test LF.used_bytes(cache) <= capacity

                unread_present = count(i -> LF.get(cache, "ns", UInt8[1, i]) !== nothing, 1:20)
                read_present   = count(i -> LF.get(cache, "ns", UInt8[1, i]) !== nothing, 21:40)
                new_present    = count(i -> LF.get(cache, "ns", UInt8[2, i]) !== nothing, 1:80)
                @test unread_present + read_present + new_present < 120
                # Never-read old entries are the first to go.
                @test unread_present == 0
                # Read entries outlive never-read newer writes.
                @test read_present == 20
                @test new_present < 80
                # Within the never-read new writes, the oldest went first.
                oldest_new = count(i -> LF.get(cache, "ns", UInt8[2, i]) !== nothing, 1:20)
                newest_new = count(i -> LF.get(cache, "ns", UInt8[2, i]) !== nothing, 61:80)
                @test oldest_new < newest_new
                @test newest_new == 20
            finally
                LF.close(cache)
            end
        end
    end

    @testset "eviction frees to three quarters of the capacity" begin
        mktempdir() do dir
            capacity = 4 * MiB
            cache = LF.open(dir; capacity, clock=FakeClock(0, 1))
            try
                i = 0
                while LF.used_bytes(cache) <= 0.9 * capacity
                    i += 1
                    LF.put!(cache, "ns", UInt8[i % 256, i ÷ 256], rand(UInt8, 50_000))
                end
                before = LF.used_bytes(cache)
                # Below the capacity nothing is evicted ...
                @test LF.maybe_evict_lru!(cache, 100_000)
                @test LF.used_bytes(cache) == before
                # ... crossing it prunes down to three quarters, minus the requested room.
                @test LF.maybe_evict_lru!(cache, 600_000)
                after = LF.used_bytes(cache)
                @test after < before
                @test after + 600_000 <= capacity * 3 ÷ 4
                # Room for a large incoming value is made too.
                large = rand(UInt8, 900_000)
                @test LF.put!(cache, "ns", b"large", large)
                @test LF.get(cache, "ns", b"large") == large
                @test LF.used_bytes(cache) <= capacity
            finally
                LF.close(cache)
            end
        end
    end

    @testset "access time refresh and granularity" begin
        mktempdir() do dir
            clock = FakeClock(1000, 0)   # frozen
            cache = LF.open(dir; capacity=4 * MiB, clock)
            try
                LMDB = LF.LMDB
                atime(k) = LMDB.Transaction(cache.env; flags=LMDB.MDB_RDONLY) do txn
                    LMDB.get(txn, cache.objmeta, LF.objkey(LF.entry_hash("ns", k)), Int64)
                end
                LF.put!(cache, "ns", b"k", b"v")
                @test atime(b"k") == 1000
                LF.get(cache, "ns", b"k")
                @test atime(b"k") == 1000 | LF.HIT_BIT          # first hit always refreshes
                clock.t += LF.ATIME_GRANULARITY - 1
                LF.get(cache, "ns", b"k")
                @test atime(b"k") == 1000 | LF.HIT_BIT          # within the window: unchanged
                clock.t += 1
                LF.get(cache, "ns", b"k")
                @test atime(b"k") == (1000 + LF.ATIME_GRANULARITY) | LF.HIT_BIT
                # Exactly one index entry per object.
                @test LMDB.Transaction(cache.env; flags=LMDB.MDB_RDONLY) do txn
                    LMDB.stat(txn, cache.objmeta).entries
                end == 3   # schema + objkey atime + one metakey
            finally
                LF.close(cache)
            end
        end
    end

    @testset "cross-session: hits made in an earlier session protect entries" begin
        mktempdir() do dir
            capacity = 4 * MiB
            clock = FakeClock(0, 1)
            keys = [UInt8[1, i] for i in 1:25]
            cache = LF.open(dir; capacity, clock)
            for k in keys
                LF.put!(cache, "ns", k, rand(UInt8, 50_000))
            end
            LF.close(cache)

            hot, cold = keys[1:12], keys[13:end]
            cache = LF.open(dir; capacity, clock)
            for k in hot
                @test LF.get(cache, "ns", k) !== nothing
            end
            LF.close(cache)

            cache = LF.open(dir; capacity, clock)
            try
                for i in 1:60
                    LF.put!(cache, "ns", UInt8[2, i], rand(UInt8, 50_000))
                end
                hot_present  = count(k -> LF.get(cache, "ns", k) !== nothing, hot)
                cold_present = count(k -> LF.get(cache, "ns", k) !== nothing, cold)
                @test hot_present == length(hot)
                @test cold_present == 0
            finally
                LF.close(cache)
            end
        end
    end

    @testset "store errors degrade to misses" begin
        mktempdir() do dir
            store = LF.LMDBStore()
            withenv("JULIA_OBJCACHE_PATH" => dir) do
                @test OC.store_enabled(store)
                LF.close(store.cache)   # simulate a broken env
                @test OC.store_get(store, "ns", UInt8[1]) === nothing
                @test !OC.store_put!(store, "ns", UInt8[1], UInt8[1])
            end
        end
    end

    @testset "process-level: capacity and location honoured" begin
        mktempdir() do dir
            ok, out = @with_objcache dir ["JULIA_OBJCACHE_CAPACITY" => string(8 * MiB)] begin
                @test OC.enabled()
                k = OC.keyhash(1, "located")
                @test OC.put!(NS, k, UInt8[1, 2, 3])
                @test OC.get(NS, k) == UInt8[1, 2, 3]   # synchronous: no polling needed
                @test isfile(joinpath(ENV["JULIA_OBJCACHE_PATH"], "fallback-lmdb1", "data.mdb"))
                c = OC.LMDBFallback.STORE.cache
                @test c.capacity == 8 * 1024 * 1024
                @test OC.LMDBFallback.LMDB.info(c.env).mapsize == 16 * 1024 * 1024
            end
            check_child(ok, out)
        end
    end
end

end # !HAS_KV_API
