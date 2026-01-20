using Test
include("helpers.jl")

precompile_test_harness("Inference caching") do load_path
    write(joinpath(load_path, "ExampleCompiler.jl"), :(module ExampleCompiler
        using CompilerCaching

        const CC = Core.Compiler

        struct ExampleInterpreter <: CC.AbstractInterpreter
            world::UInt
            cache::CacheHandle
            inf_cache::Vector{CC.InferenceResult}
        end
        ExampleInterpreter(cache::CacheHandle, world::UInt) =
            ExampleInterpreter(world, cache, CC.InferenceResult[])

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

        codegen_count = Ref(0)
        function infer(cache, mi, world)
            interp = ExampleInterpreter(cache, world)
            populate!(cache, interp, mi)
        end
        codegen(cache, mi, world, codeinfos) = (codegen_count[] += 1; :codegen_result)
        link(cache, mi, world, ir_data) = (ir_data)

        function precompile(f, tt)
            world = Base.get_world_counter()
            mi = method_instance(f, tt; world)
            cache = CacheHandle(:ExampleCompiler)
            result = cached_compilation(cache, mi, world;
                infer=infer, codegen=codegen, link=link)
            @assert result === :codegen_result
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
        @test ExampleCompiler.codegen_count[] == 0

        cache = CacheHandle(:ExampleCompiler)

        # Check that no cached entry is present
        identity_mi = method_instance(identity, (Int,))
        @test check_presence(identity_mi, cache) === nothing

        using ExampleUser
        @test ExampleCompiler.codegen_count[] == 0

        # Check that kernel survived
        square_mi = method_instance(ExampleUser.square, (Float64,))
        @test check_presence(square_mi, cache) !== nothing
        ExampleCompiler.precompile(ExampleUser.square, (Float64,))
        @test ExampleCompiler.codegen_count[] == 0

        # check that identity survived
        @test check_presence(identity_mi, cache) !== nothing broken=VERSION>=v"1.12.0-DEV.1268"
        ExampleCompiler.precompile(identity, (Int,))
        @test ExampleCompiler.codegen_count[] == 0 broken=VERSION>=v"1.12.0-DEV.1268"
    end
end
