using Test, CompilerCaching

function precompile_test_harness(@nospecialize(f), testset::String)
    @testset "$testset" begin
        precompile_test_harness(f, true)
    end
end
function precompile_test_harness(@nospecialize(f), separate::Bool)
    # XXX: clean-up may fail on Windows, because opened files are not deletable.
    #      fix this by running the harness in a separate process, such that the
    #      compilation cache files are not opened?
    load_path = mktempdir(cleanup=true)
    load_cache_path = separate ? mktempdir(cleanup=true) : load_path
    try
        pushfirst!(LOAD_PATH, load_path)
        pushfirst!(DEPOT_PATH, load_cache_path)
        f(load_path)
    finally
        popfirst!(DEPOT_PATH)
        popfirst!(LOAD_PATH)
    end
    nothing
end

precompile_test_harness("Inference caching") do load_path
    write(joinpath(load_path, "ExampleCompiler.jl"), :(module ExampleCompiler
        using CompilerCaching

        const CC = Core.Compiler

        struct ExampleInterpreter <: CC.AbstractInterpreter
            world::UInt
            cache::CacheView
            inf_cache::Vector{CC.InferenceResult}
        end
        ExampleInterpreter(cache::CacheView) =
            ExampleInterpreter(cache.world, cache, CC.InferenceResult[])

        CC.InferenceParams(::ExampleInterpreter) = CC.InferenceParams()
        CC.OptimizationParams(::ExampleInterpreter) = CC.OptimizationParams()
        CC.get_inference_cache(interp::ExampleInterpreter) = interp.inf_cache
        @static if isdefined(Core.Compiler, :get_inference_world)
            Core.Compiler.get_inference_world(interp::ExampleInterpreter) = interp.world
        else
            Core.Compiler.get_world_counter(interp::ExampleInterpreter) = interp.world
        end
        CC.lock_mi_inference(::ExampleInterpreter, ::Core.MethodInstance) = nothing
        CC.unlock_mi_inference(::ExampleInterpreter, ::Core.MethodInstance) = nothing
        @setup_caching ExampleInterpreter.cache

        emit_code_count = Ref(0)
        function emit_ir(cache, mi)
            interp = ExampleInterpreter(cache)
            typeinf!(cache, interp, mi)
        end
        emit_code(cache, mi, ir) = (emit_code_count[] += 1; :code_result)
        emit_executable(cache, mi, code) = code

        function precompile(f, tt)
            world = Base.get_world_counter()
            mi = method_instance(f, tt; world)
            cache = CacheView(:ExampleCompiler, world)
            result = get!(cache, mi, :executable) do cache, mi
                ir = get!(emit_ir, cache, mi, :ir)
                code = get!(cache, mi, :code) do cache, mi
                    emit_code(cache, mi, ir)
                end
                emit_executable(cache, mi, code)
            end
            @assert result === :code_result
        end

        end # module
    ) |> string)
    Base.compilecache(Base.PkgId("ExampleCompiler"), stderr, stdout)

    write(joinpath(load_path, "ExampleUser.jl"), :(module ExampleUser
        import ExampleCompiler
        using PrecompileTools

        function square(x)
            return x*x
        end

        ExampleCompiler.precompile(square, (Float64,))

        # identity is foreign
        @setup_workload begin
            @compile_workload begin
                ExampleCompiler.precompile(identity, (Int64,))
            end
        end
        end# module
    ) |> string)

    Base.compilecache(Base.PkgId("ExampleUser"), stderr, stdout)
    @eval let
        using CompilerCaching
        import ExampleCompiler
        @test ExampleCompiler.emit_code_count[] == 0

        cache = CacheView(:ExampleCompiler, Base.get_world_counter())

        # Check that no cached entry is present
        identity_mi = method_instance(identity, (Int,))
        @test !haskey(cache, identity_mi)

        using ExampleUser
        @test ExampleCompiler.emit_code_count[] == 0

        # importing the package bumps the world age, so get a new cache view
        cache = CacheView(:ExampleCompiler, Base.get_world_counter())

        # Check that kernel survived
        square_mi = method_instance(ExampleUser.square, (Float64,))
        @test haskey(cache, square_mi)
        ExampleCompiler.precompile(ExampleUser.square, (Float64,))
        @test ExampleCompiler.emit_code_count[] == 0

        # check that identity survived
        @test haskey(cache, identity_mi) broken=VERSION>=v"1.12.0-DEV.1268"
        ExampleCompiler.precompile(identity, (Int,))
        @test ExampleCompiler.emit_code_count[] == 0 broken=VERSION>=v"1.12.0-DEV.1268"
    end
end
