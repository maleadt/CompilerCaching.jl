# Tests for CompilerCaching.ObjCache.
#
# Anything that touches a store runs in a child process with JULIA_OBJCACHE_PATH set to
# a fresh temporary directory, so the developer's depot is never written to. Tests use
# the runtime backend's weaker asynchronous contract: children poll `get` (up to 5 s)
# before asserting a hit, and a producing child polls for the commit before exiting.

using CompilerCaching
using Serialization
using Test

const OC = CompilerCaching.ObjCache

"""
Evaluate `expr` in a child Julia with its object cache isolated at `objcache_path`.
Returns `(success, output)`.
"""
function run_objcache_child(expr::Expr, objcache_path::AbstractString; env=())
    project = dirname(Base.active_project())
    runner = joinpath(@__DIR__, "objcache_child.jl")
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$project $runner`
    cmd = addenv(cmd, "JULIA_OBJCACHE_PATH" => objcache_path, env...)
    input = IOBuffer()
    serialize(input, expr)
    seekstart(input)
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(cmd); stdin=input, stdout=out, stderr=out))
    return success(proc), String(take!(out))
end

# Like Distributed.@everywhere, capture code as an expression and allow `$`
# interpolation from the calling scope. With no path, use a fresh temporary cache.
macro with_objcache(expr)
    quote
        mktempdir() do objcache_path
            run_objcache_child($(esc(Expr(:quote, expr))), objcache_path)
        end
    end
end

macro with_objcache(objcache_path, expr)
    quote
        run_objcache_child($(esc(Expr(:quote, expr))), $(esc(objcache_path)))
    end
end

macro with_objcache(objcache_path, env, expr)
    quote
        run_objcache_child($(esc(Expr(:quote, expr))), $(esc(objcache_path)); env=$(esc(env)))
    end
end

# Report a child's outcome; print its output when it failed so the failure is legible.
function check_child(ok::Bool, out::AbstractString)
    ok || println(stderr, "--- child output ---\n", out, "\n--- end ---")
    @test ok
    return ok
end

child_enabled(out) = occursin("ENABLED=true", out)

@testset "ObjCache" begin

@testset "keyhash" begin
    # Framing: u32be(schema) then u32be(len) ‖ bytes per field.
    @test OC.keyhash(0) == hex2bytes("df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119")  # sha256 of 4 zero bytes
    @test bytes2hex(OC.keyhash(4, "id", v"12.0", 3, UInt8[1, 2, 3])) ==
          "57ae358f3cec40426cbb85414a9300ea854f83488278267a24dca25ada380992"
    @test length(OC.keyhash(1, "x")) == 32

    # Injectivity of the framing.
    @test OC.keyhash(1, "ab", "c") != OC.keyhash(1, "a", "bc")
    @test OC.keyhash(1, "abc") != OC.keyhash(1, "ab", "c")
    @test OC.keyhash(1, "") != OC.keyhash(1)
    @test OC.keyhash(1, "a") != OC.keyhash(2, "a")

    # Field encodings: strings by code units, integers/versions by their string form.
    @test OC.keyhash(1, "abc") == OC.keyhash(1, codeunits("abc")) == OC.keyhash(1, UInt8['a', 'b', 'c'])
    @test OC.keyhash(1, 42) == OC.keyhash(1, "42") == OC.keyhash(1, UInt8(42))
    @test OC.keyhash(1, v"1.2.3") == OC.keyhash(1, "1.2.3")
    @test OC.keyhash(1, SubString("xabcx", 2, 4)) == OC.keyhash(1, "abc")
    @test OC.keyhash(1, view(UInt8[0, 1, 2, 3], 2:3)) == OC.keyhash(1, UInt8[1, 2])

    @test_throws ArgumentError OC.keyhash(-1, "a")
    @test_throws ArgumentError OC.keyhash(1, 1.5)
    @test_throws ArgumentError OC.keyhash(1, nothing)
end

@testset "disabled (JULIA_OBJCACHE=0)" begin
    mktempdir() do dir
        ok, out = @with_objcache dir ["JULIA_OBJCACHE" => "0"] begin
            @testset "disabled" begin
                @test !OC.enabled()
                k = OC.keyhash(1, "disabled")
                @test OC.put!(NS, k, UInt8[1, 2, 3]) === false
                @test OC.get(NS, k) === nothing
                calls = Ref(0)
                for _ in 1:3
                    r = OC.get!(NS, "disabled", "field"; schema=1, persistable=true) do
                        calls[] += 1
                        UInt8[calls[]]
                    end
                    @test r == UInt8[calls[]]
                end
                @test calls[] == 3
                @test isempty(readdir(ENV["JULIA_OBJCACHE_PATH"]))
            end
        end
        check_child(ok, out)
    end
end

@testset "store round trip" begin
    ok, out = @with_objcache begin
        println("ENABLED=", OC.enabled())
        if OC.enabled()
            if OC.HAS_KV_API
                @test all(pkgid -> pkgid.name != "LMDB", keys(Base.loaded_modules))
            end
            @testset "round trip" begin
                value = rand(UInt8, 1000)
                key = OC.keyhash(1, "roundtrip")
                @test OC.get(NS, key) === nothing
                @test OC.put!(NS, key, value)
                @test poll_get(NS, key) == value
                @test OC.get(NS * "/other", key) === nothing

                key = OC.keyhash(1, "roundtrip-views")
                @test OC.put!(NS, view(key, 1:32), view(value, 1:10))
                @test poll_get(NS, key) == value[1:10]
                @test OC.get!(() -> UInt8[9], NS, view(key, 1:32); persistable=true) == value[1:10]
                @test_throws ArgumentError OC.get(NS * "\0bad", key)
                @test_throws ArgumentError OC.put!(NS * "\0bad", key, value)
            end

            @testset "get! read-through" begin
                calls = Ref(0)
                compile = () -> (calls[] += 1; UInt8[0xca, 0xfe])
                value = OC.get!(compile, NS, "getbang", v"1.0", 7;
                                schema=1, persistable=true)
                @test value == UInt8[0xca, 0xfe]
                @test calls[] == 1
                @test poll_get(NS, OC.keyhash(1, "getbang", v"1.0", 7)) == value
                @test OC.get!(compile, NS, "getbang", v"1.0", 7;
                              schema=1, persistable=true) == value
                @test calls[] == 1

                key = OC.keyhash(1, "raw")
                @test OC.get!(() -> UInt8[1], NS, key; persistable=true) == UInt8[1]
                @test poll_get(NS, key) == UInt8[1]
                @test OC.get!(() -> UInt8[2], NS, key; persistable=true) == UInt8[1]
                @test OC.get!(() -> UInt8[3], NS, key;
                              schema=1, persistable=true) == UInt8[3]
                @test poll_get(NS, OC.keyhash(1, key)) == UInt8[3]
                @test_throws ArgumentError OC.get!(() -> UInt8[4], NS * "\0bad", key;
                                                   persistable=true)
                # A disabled persistence policy does not inspect store inputs.
                @test OC.get!(() -> UInt8[4], NS * "\0bad", key;
                              persistable=false) == UInt8[4]
                @test_throws ArgumentError OC.get!(() -> "nope", NS, "badtype";
                                                   schema=1, persistable=true)
            end

            @testset "persistable=false never touches the store" begin
                key = OC.keyhash(1, "nonpersistable")
                calls = Ref(0)
                for _ in 1:2
                    value = OC.get!(NS, "nonpersistable"; schema=1, persistable=false) do
                        calls[] += 1
                        UInt8[7]
                    end
                    @test value == UInt8[7]
                end
                @test calls[] == 2
                settle("nonpersistable")
                @test OC.get(NS, key) === nothing
                # Even with the key pre-populated, persistable=false does not read.
                OC.put!(NS, key, UInt8[8])
                @test poll_get(NS, key) == UInt8[8]
                @test OC.get!(() -> UInt8[9], NS, "nonpersistable"; schema=1, persistable=false) == UInt8[9]
            end

            @testset "exceptions propagate without storing" begin
                key = OC.keyhash(1, "throws")
                @test_throws ErrorException OC.get!(NS, "throws"; schema=1, persistable=true) do
                    error("compile failed")
                end
                settle("throws")
                @test OC.get(NS, key) === nothing
            end
        end
    end
    check_child(ok, out)
    # A store is available on every supported Julia: the runtime's, or the fallback.
    @test child_enabled(out)
end

@testset "cross-process persistence" begin
    mktempdir() do dir
        token = string(rand(UInt64); base=16)
        ok, out = @with_objcache dir begin
            println("ENABLED=", OC.enabled())
            if OC.enabled()
                k = OC.keyhash(1, "crossproc", $token)
                @test OC.put!(NS, k, codeunits("payload-" * $token))
                # Poll for the commit before exiting: queued writes are not drained at exit.
                @test poll_get(NS, k) == codeunits("payload-" * $token)
            end
        end
        check_child(ok, out)
        if child_enabled(out)
            ok, out = @with_objcache dir begin
                @test OC.enabled()
                k = OC.keyhash(1, "crossproc", $token)
                @test OC.get(NS, k) == codeunits("payload-" * $token)
                calls = Ref(0)
                r = OC.get!(NS, "crossproc", $token; schema=1, persistable=true) do
                    calls[] += 1
                    UInt8[]
                end
                @test r == codeunits("payload-" * $token)
                @test calls[] == 0
            end
            check_child(ok, out)
        end
    end
end

end # testset ObjCache
