using CompilerCaching
using Test
using Base.Experimental: @MethodTable

include("utils.jl")

@testset "CompilerCaching" begin

#==============================================================================#
# Mode 1: Custom IR - Empty methods with add_method
#==============================================================================#

@testset "Mode 1: Custom IR" begin

@testset "basic caching" begin
    method_table = @eval @MethodTable $(gensym(:method_table))
    cache = CompilerCache(:BasicTest, method_table)

    function basic_node end
    add_method(cache, basic_node, (Int,), 10)

    compile_count = Ref(0)
    function my_compile(mi)
        source = mi.def.source
        compile_count[] += 1
        source * 2
    end

    world = Base.get_world_counter()
    mi = method_instance(basic_node, (Int,); world, method_table=cache.method_table)

    # First call: cache miss, compile_fn invoked
    r1 = cached_compilation(cache, mi, world) do ctx
        my_compile(mi)
    end
    @test r1 == 20  # 10 * 2
    @test compile_count[] == 1

    # Second call: cache hit, compile_fn NOT invoked
    r2 = cached_compilation(cache, mi, world) do ctx
        my_compile(mi)
    end
    @test r2 == 20
    @test compile_count[] == 1  # still 1

    # Redefine method → invalidates cache, recompile
    add_method(cache, basic_node, (Int,), 30)
    world = Base.get_world_counter()
    mi = method_instance(basic_node, (Int,); world, method_table=cache.method_table)
    r3 = cached_compilation(cache, mi, world) do ctx
        my_compile(mi)
    end
    @test r3 == 60  # 30 * 2
    @test compile_count[] == 2  # incremented
end

@testset "multiple dispatch" begin
    method_table = @eval @MethodTable $(gensym(:method_table))
    cache = CompilerCache(:DispatchTest, method_table)

    function dispatch_node end
    add_method(cache, dispatch_node, (Int,), 100)
    add_method(cache, dispatch_node, (Float64,), 200)

    compile_count = Ref(0)
    function my_compile(mi)
        source = mi.def.source
        compile_count[] += 1
        source + 1
    end

    world = Base.get_world_counter()
    mi_int = method_instance(dispatch_node, (Int,); world, method_table=cache.method_table)
    mi_float = method_instance(dispatch_node, (Float64,); world, method_table=cache.method_table)

    # Different types → different cache entries, each compiles once
    r_int = cached_compilation(cache, mi_int, world) do ctx
        my_compile(mi_int)
    end
    @test r_int == 101
    @test compile_count[] == 1

    r_float = cached_compilation(cache, mi_float, world) do ctx
        my_compile(mi_float)
    end
    @test r_float == 201
    @test compile_count[] == 2

    # Cache hits - no recompilation
    r_int2 = cached_compilation(cache, mi_int, world) do ctx
        my_compile(mi_int)
    end
    @test r_int2 == 101
    @test compile_count[] == 2  # unchanged

    r_float2 = cached_compilation(cache, mi_float, world) do ctx
        my_compile(mi_float)
    end
    @test r_float2 == 201
    @test compile_count[] == 2  # unchanged

    # Redefine only Int method → only Int recompiles
    add_method(cache, dispatch_node, (Int,), 50)
    world = Base.get_world_counter()
    mi_int = method_instance(dispatch_node, (Int,); world, method_table=cache.method_table)
    r_int3 = cached_compilation(cache, mi_int, world) do ctx
        my_compile(mi_int)
    end
    @test r_int3 == 51
    @test compile_count[] == 3

    # Float64 still uses cached version (need to re-lookup mi after world change)
    mi_float = method_instance(dispatch_node, (Float64,); world, method_table=cache.method_table)
    r_float3 = cached_compilation(cache, mi_float, world) do ctx
        my_compile(mi_float)
    end
    @test r_float3 == 201
    @test compile_count[] == 3  # unchanged
end

@testset "nothing on missing method" begin
    method_table = @eval @MethodTable $(gensym(:method_table))
    cache = CompilerCache(:MissingTest, method_table)

    function missing_node end
    # No method registered

    # Returns nothing when no method found
    world = Base.get_world_counter()
    mi = method_instance(missing_node, (Int,); world, method_table=cache.method_table)
    @test mi === nothing
end

@testset "sharding keys" begin
    # Compiler with NamedTuple keys
    ShardKeys = @NamedTuple{opt_level::Int, debug::Bool}
    method_table = @eval @MethodTable $(gensym(:method_table))
    cache = CompilerCache{ShardKeys}(:ShardingTest, method_table)

    function sharded_node end
    add_method(cache, sharded_node, (Int,), 42)

    compile_count = Ref(0)
    function my_compile(mi)
        compile_count[] += 1
        mi.def.source
    end

    world = Base.get_world_counter()
    mi = method_instance(sharded_node, (Int,); world, method_table=cache.method_table)

    # First key combination
    keys1 = (opt_level=1, debug=false)
    r1 = cached_compilation(cache, mi, world, keys1) do ctx
        my_compile(mi)
    end
    @test r1 == 42
    @test compile_count[] == 1

    # Same key combination → cache hit
    r2 = cached_compilation(cache, mi, world, keys1) do ctx
        my_compile(mi)
    end
    @test r2 == 42
    @test compile_count[] == 1  # unchanged

    # Different key combination → cache miss (different shard)
    keys2 = (opt_level=2, debug=false)
    r3 = cached_compilation(cache, mi, world, keys2) do ctx
        my_compile(mi)
    end
    @test r3 == 42
    @test compile_count[] == 2  # new compilation

    # Yet another key combination
    keys3 = (opt_level=1, debug=true)
    r4 = cached_compilation(cache, mi, world, keys3) do ctx
        my_compile(mi)
    end
    @test r4 == 42
    @test compile_count[] == 3  # another new compilation

    # Back to first key combination → still cached
    r5 = cached_compilation(cache, mi, world, keys1) do ctx
        my_compile(mi)
    end
    @test r5 == 42
    @test compile_count[] == 3  # unchanged
end

@testset "CompilationContext for dependencies" begin
    method_table = @eval @MethodTable $(gensym(:method_table))
    cache = CompilerCache(:CtxTest, method_table)

    function ctx_node end
    add_method(cache, ctx_node, (Int,), :my_source)

    captured_source = Ref{Any}(nothing)

    world = Base.get_world_counter()
    mi = method_instance(ctx_node, (Int,); world, method_table=cache.method_table)

    cached_compilation(cache, mi, world) do ctx
        captured_source[] = mi.def.source
        :compiled
    end

    @test captured_source[] === :my_source
    @test mi isa Core.MethodInstance
    @test mi.def.name === :ctx_node
end

@testset "dependency invalidation" begin
    method_table = @eval @MethodTable $(gensym(:method_table))
    cache = CompilerCache(:DepTest, method_table)

    function parent_node end
    function child_node end
    add_method(cache, child_node, (Int,), :child_ir)
    add_method(cache, parent_node, (Int,), :parent_ir)

    child_compile_count = Ref(0)
    parent_compile_count = Ref(0)

    function child_compile(mi)
        child_compile_count[] += 1
        :child_compiled
    end

    function parent_compile(ctx, mi, world)
        parent_compile_count[] += 1
        # Register dependency on child
        child_mi = method_instance(child_node, (Int,); world, method_table=cache.method_table)
        register_dependency!(ctx, child_mi)
        :parent_compiled
    end

    world = Base.get_world_counter()
    child_mi = method_instance(child_node, (Int,); world, method_table=cache.method_table)
    parent_mi = method_instance(parent_node, (Int,); world, method_table=cache.method_table)

    # Compile child first
    cached_compilation(cache, child_mi, world) do ctx
        child_compile(child_mi)
    end
    @test child_compile_count[] == 1

    # Compile parent (depends on child)
    cached_compilation(cache, parent_mi, world) do ctx
        parent_compile(ctx, parent_mi, world)
    end
    @test parent_compile_count[] == 1

    # Cache hits
    cached_compilation(cache, child_mi, world) do ctx
        child_compile(child_mi)
    end
    cached_compilation(cache, parent_mi, world) do ctx
        parent_compile(ctx, parent_mi, world)
    end
    @test child_compile_count[] == 1
    @test parent_compile_count[] == 1

    # Redefine child → child recompiles
    add_method(cache, child_node, (Int,), :new_child_ir)
    world = Base.get_world_counter()
    child_mi = method_instance(child_node, (Int,); world, method_table=cache.method_table)
    cached_compilation(cache, child_mi, world) do ctx
        child_compile(child_mi)
    end
    @test child_compile_count[] == 2

    # Parent should also recompile due to dependency
    parent_mi = method_instance(parent_node, (Int,); world, method_table=cache.method_table)
    cached_compilation(cache, parent_mi, world) do ctx
        parent_compile(ctx, parent_mi, world)
    end
    @test parent_compile_count[] == 2
end

@testset "method table isolation" begin
    method_table_a = @eval @MethodTable $(gensym(:method_table_a))
    cache_a = CompilerCache(:IsolationA, method_table_a)
    method_table_b = @eval @MethodTable $(gensym(:method_table_b))
    cache_b = CompilerCache(:IsolationB, method_table_b)

    function isolated_node end
    add_method(cache_a, isolated_node, (Int,), :ir_a)
    add_method(cache_b, isolated_node, (Int,), :ir_b)

    result_a = Ref{Any}(nothing)
    result_b = Ref{Any}(nothing)

    world = Base.get_world_counter()
    mi_a = method_instance(isolated_node, (Int,); world, method_table=cache_a.method_table)
    mi_b = method_instance(isolated_node, (Int,); world, method_table=cache_b.method_table)

    cached_compilation(cache_a, mi_a, world) do ctx
        result_a[] = mi_a.def.source
        mi_a.def.source
    end
    cached_compilation(cache_b, mi_b, world) do ctx
        result_b[] = mi_b.def.source
        mi_b.def.source
    end

    @test result_a[] === :ir_a
    @test result_b[] === :ir_b
end

@testset "complex source types" begin
    method_table = @eval @MethodTable $(gensym(:method_table))
    cache = CompilerCache(:ComplexTest, method_table)

    # Store a complex struct as source
    struct MyIR
        nodes::Vector{Symbol}
        edges::Dict{Symbol, Vector{Symbol}}
    end

    function complex_node end
    ir = MyIR([:a, :b, :c], Dict(:a => [:b], :b => [:c]))
    add_method(cache, complex_node, (Int,), ir)

    captured_ir = Ref{Any}(nothing)

    world = Base.get_world_counter()
    mi = method_instance(complex_node, (Int,); world, method_table=cache.method_table)

    result = cached_compilation(cache, mi, world) do ctx
        source = mi.def.source
        captured_ir[] = source
        length(source.nodes)
    end
    @test result == 3
    @test captured_ir[] isa MyIR
    @test captured_ir[].nodes == [:a, :b, :c]
    @test captured_ir[].edges[:a] == [:b]
end

end # Mode 1

#==============================================================================#
# Mode 2: Overlay Methods - Julia source in custom MT
#==============================================================================#

@testset "Mode 2: Overlay Methods" begin

@testset "Julia method in custom MT" begin
    # overlay_double is defined at top level with @overlay
    method_table = @eval @MethodTable $(gensym(:method_table))
    overlay_double_name = gensym("overlay_double")
    overlay_double = @eval begin
        function $overlay_double_name end
        Base.Experimental.@overlay $method_table function $overlay_double_name(x::Int)
            x * 2
        end
        $overlay_double_name
    end
    cache = CompilerCache(:OverlayTest, method_table)

    compile_count = Ref(0)
    function my_compile(mi)
        compile_count[] += 1
        # Source is compressed Julia source (can use Base.uncompressed_ast if needed)
        # Return something based on the method
        mi.def.name
    end

    world = Base.get_world_counter()
    mi = method_instance(overlay_double, (Int,); world, method_table=cache.method_table)

    result = cached_compilation(cache, mi, world) do ctx
        my_compile(mi)
    end
    # Overlay methods may have gensym'd names like "#overlay_double"
    @test occursin("overlay_double", string(result))
    @test compile_count[] == 1

    # Cache hit
    result2 = cached_compilation(cache, mi, world) do ctx
        my_compile(mi)
    end
    @test occursin("overlay_double", string(result2))
    @test compile_count[] == 1  # unchanged
end

end # Mode 2

#==============================================================================#
# Mode 3: Global Methods - Julia source in global MT
#==============================================================================#

@testset "Mode 3: Global Methods" begin

@testset "global MT lookup" begin
    # Use Compiler without method table = global MT
    cache = CompilerCache(:GlobalTest)

    # Define a regular Julia function (in global MT)
    global_test_fn(x::Int) = x + 100

    compile_count = Ref(0)
    function my_compile(mi)
        compile_count[] += 1
        # For global methods, source may be compressed (String/Vector{UInt8})
        # Use Base.uncompressed_ast() if you need CodeInfo
        # Here we just return the method name
        mi.def.name
    end

    world = Base.get_world_counter()
    mi = method_instance(global_test_fn, (Int,); world, method_table=cache.method_table)

    result = cached_compilation(cache, mi, world) do ctx
        my_compile(mi)
    end
    @test result === :global_test_fn
    @test compile_count[] == 1

    # Cache hit
    result2 = cached_compilation(cache, mi, world) do ctx
        my_compile(mi)
    end
    @test result2 === :global_test_fn
    @test compile_count[] == 1  # unchanged
end

@testset "global MT with sharding keys" begin
    # Global MT with sharding keys (e.g., for GPUCompiler-style usage)
    ShardKeys = @NamedTuple{opt_level::Int}
    cache = CompilerCache{ShardKeys}(:GlobalShardTest)

    global_sharded_fn(x::Float64) = x * 2.0

    compile_count = Ref(0)
    function my_compile(mi)
        compile_count[] += 1
        mi.def.name
    end

    world = Base.get_world_counter()
    mi = method_instance(global_sharded_fn, (Float64,); world, method_table=cache.method_table)

    # Different sharding keys = different cache entries
    r1 = cached_compilation(cache, mi, world, (opt_level=1,)) do ctx
        my_compile(mi)
    end
    @test r1 === :global_sharded_fn
    @test compile_count[] == 1

    r2 = cached_compilation(cache, mi, world, (opt_level=2,)) do ctx
        my_compile(mi)
    end
    @test r2 === :global_sharded_fn
    @test compile_count[] == 2  # different shard

    r3 = cached_compilation(cache, mi, world, (opt_level=1,)) do ctx
        my_compile(mi)
    end
    @test r3 === :global_sharded_fn
    @test compile_count[] == 2  # cache hit
end

end # Mode 3

#==============================================================================#
# populate!
#==============================================================================#

@testset "populate!" begin
    # Simple test interpreter
    struct TestInterpreter <: Core.Compiler.AbstractInterpreter
        world::UInt
        inf_cache::Vector{Core.Compiler.InferenceResult}
    end
    TestInterpreter(world::UInt) = TestInterpreter(world, Core.Compiler.InferenceResult[])

    Core.Compiler.InferenceParams(::TestInterpreter) = Core.Compiler.InferenceParams()
    Core.Compiler.OptimizationParams(::TestInterpreter) = Core.Compiler.OptimizationParams()
    Core.Compiler.get_inference_cache(interp::TestInterpreter) = interp.inf_cache
    @static if isdefined(Core.Compiler, :get_inference_world)
        Core.Compiler.get_inference_world(interp::TestInterpreter) = interp.world
    else
        Core.Compiler.get_world_counter(interp::TestInterpreter) = interp.world
    end
    Core.Compiler.cache_owner(::TestInterpreter) = :test_interp
    Core.Compiler.lock_mi_inference(::TestInterpreter, ::Core.MethodInstance) = nothing
    Core.Compiler.unlock_mi_inference(::TestInterpreter, ::Core.MethodInstance) = nothing

    @testset "basic inference" begin
        cache = CompilerCache(:InferenceTest)
        test_fn(x::Int) = x + 1
        world = Base.get_world_counter()
        mi = method_instance(test_fn, (Int,); world)

        interp = TestInterpreter(world)
        result = populate!(cache, interp, mi)

        @static if VERSION >= v"1.12.0-DEV.15"
            @test result isa Vector{Pair{Core.CodeInstance, Core.CodeInfo}}
        else
            @test result === nothing
        end
    end
end

end # @testset "ForeignCompiler"

println("All tests passed!")
