using CompilerCaching
using Test
using Base.Experimental: @MethodTable

# Common results struct for tests
mutable struct TestResults
    result::Any
    TestResults() = new(nothing)
end

@testset "CompilerCaching" verbose=true begin

@testset "basic caching" begin
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
    cache = CacheView{TestResults}(:GlobalTest, world)
    mi = method_instance(global_test_fn, (Int,); world)

    # First call - cache miss, compile
    ci = get!(cache, mi) do
        create_ci(cache, mi)
    end
    res = results(cache, ci)
    @test res.result === nothing
    res.result = my_compile(mi)
    @test res.result === :global_test_fn
    @test compile_count[] == 1

    # Cache hit - same CI returned
    ci2 = get!(cache, mi) do
        create_ci(cache, mi)
    end
    res2 = results(cache, ci2)
    @test res2.result === :global_test_fn
    @test ci2 === ci
    @test res2 === res
    @test compile_count[] == 1  # unchanged
end

@testset "cache partitioning" begin
    # Global MT with sharding keys (e.g., for GPUCompiler-style usage)
    # Different caches for different key combinations
    global_sharded_fn(x::Float64) = x * 2.0

    compile_count = Ref(0)
    function my_compile(mi)
        compile_count[] += 1
        mi.def.name
    end

    world = Base.get_world_counter()
    cache1 = CacheView{TestResults}((:GlobalShardTest, #=opt_level=# 1), world)
    cache2 = CacheView{TestResults}((:GlobalShardTest, #=opt_level=# 2), world)
    mi = method_instance(global_sharded_fn, (Float64,); world)

    # Different sharding keys = different cache entries
    ci1 = get!(cache1, mi) do
        create_ci(cache1, mi)
    end
    res1 = results(cache1, ci1)
    res1.result = my_compile(mi)
    @test res1.result === :global_sharded_fn
    @test compile_count[] == 1

    ci2 = get!(cache2, mi) do
        create_ci(cache2, mi)
    end
    res2 = results(cache2, ci2)
    @test res2.result === nothing  # different shard, not populated
    res2.result = my_compile(mi)
    @test compile_count[] == 2

    # Cache hit on first shard
    ci3 = get!(cache1, mi) do
        create_ci(cache1, mi)
    end
    res3 = results(cache1, ci3)
    @test res3.result === :global_sharded_fn
    @test ci3 === ci1
    @test res3 === res1
    @test compile_count[] == 2  # unchanged
end

@testset "overlay method tables" begin
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

    compile_count = Ref(0)
    function my_compile(mi)
        compile_count[] += 1
        # Source is compressed Julia source (can use Base.uncompressed_ast if needed)
        # Return something based on the method
        mi.def.name
    end

    world = Base.get_world_counter()
    cache = CacheView{TestResults}(:OverlayTest, world)
    mi = method_instance(overlay_double, (Int,); world, method_table)

    ci = get!(cache, mi) do
        create_ci(cache, mi)
    end
    res = results(cache, ci)
    res.result = my_compile(mi)
    # Overlay methods may have gensym'd names like "#overlay_double"
    @test occursin("overlay_double", string(res.result))
    @test compile_count[] == 1

    # Cache hit
    ci2 = get!(cache, mi) do
        create_ci(cache, mi)
    end
    res2 = results(cache, ci2)
    @test occursin("overlay_double", string(res2.result))
    @test ci2 === ci
    @test res2 === res
    @test compile_count[] == 1  # unchanged
end

@testset "inference integration" begin
    # Results struct for inference tests
    mutable struct InferenceResults
        ir::Any
        InferenceResults() = new(nothing)
    end

    # Test interpreter that properly integrates with cache
    struct TestInterpreter <: Core.Compiler.AbstractInterpreter
        world::UInt
        cache::CacheView
        inf_cache::Vector{Core.Compiler.InferenceResult}
    end
    TestInterpreter(cache::CacheView) =
        TestInterpreter(cache.world, cache, Core.Compiler.InferenceResult[])
    @setup_caching TestInterpreter.cache

    Core.Compiler.InferenceParams(::TestInterpreter) = Core.Compiler.InferenceParams()
    Core.Compiler.OptimizationParams(::TestInterpreter) = Core.Compiler.OptimizationParams()
    Core.Compiler.get_inference_cache(interp::TestInterpreter) = interp.inf_cache
    @static if isdefined(Core.Compiler, :get_inference_world)
        Core.Compiler.get_inference_world(interp::TestInterpreter) = interp.world
    else
        Core.Compiler.get_world_counter(interp::TestInterpreter) = interp.world
    end
    Core.Compiler.lock_mi_inference(::TestInterpreter, ::Core.MethodInstance) = nothing
    Core.Compiler.unlock_mi_inference(::TestInterpreter, ::Core.MethodInstance) = nothing

    test_fn(x::Int) = x + 1
    world = Base.get_world_counter()
    cache = CacheView{InferenceResults}(:InferenceTest, world)
    mi = method_instance(test_fn, (Int,); world)

    interp = TestInterpreter(cache)
    result = typeinf!(cache, interp, mi)

    # typeinf! now always returns Vector{Pair{CI, IR}}
    @test result isa Vector
    @test length(result) >= 1
    ci, ir = first(result)
    @test ci isa Core.CodeInstance
    @test ir isa Core.CodeInfo

    # Test that results accessor works on inferred CI
    res = results(cache, ci)
    @test res isa InferenceResults

    # Test const-return functions get results wrapper
    # These functions return a constant and skip optimization, but finish! should still be called
    const_return_fn(x::Int) = nothing  # Returns constant `nothing`
    world2 = Base.get_world_counter()
    cache2 = CacheView{InferenceResults}(:InferenceTest, world2)
    mi2 = method_instance(const_return_fn, (Int,); world=world2)

    interp2 = TestInterpreter(cache2)
    result2 = typeinf!(cache2, interp2, mi2)

    @test length(result2) >= 1
    ci2, _ = first(result2)
    @test ci2 isa Core.CodeInstance
    # Verify it's actually a const-return CI (skip under coverage as it disables const-return)
    @test Core.Compiler.use_const_api(ci2) skip=(Base.JLOptions().code_coverage > 0)
    # The key test: finish! hook should have stacked our results even for const-return
    @test results(cache2, ci2) isa InferenceResults
end

#==============================================================================#
# Custom IR
#==============================================================================#

@testset "custom IR" begin

@testset "basic caching" begin
    method_table = @eval @MethodTable $(gensym(:method_table))

    function basic_node end
    add_method(method_table, basic_node, (Int,), 10)

    compile_count = Ref(0)
    function my_compile(mi)
        source = mi.def.source
        compile_count[] += 1
        source * 2
    end

    world = Base.get_world_counter()
    cache = CacheView{TestResults}(:BasicTest, world)
    mi = method_instance(basic_node, (Int,); world, method_table)

    # First call: cache miss, compile_fn invoked
    ci = get!(cache, mi) do
        create_ci(cache, mi)
    end
    res = results(cache, ci)
    res.result = my_compile(mi)
    @test res.result == 20  # 10 * 2
    @test compile_count[] == 1

    # Second call: cache hit, compile_fn NOT invoked
    ci2 = get!(cache, mi) do
        create_ci(cache, mi)
    end
    res2 = results(cache, ci2)
    @test res2.result == 20
    @test ci2 === ci
    @test res2 === res
    @test compile_count[] == 1  # still 1

    # Redefine method → invalidates cache, recompile
    add_method(method_table, basic_node, (Int,), 30)
    world = Base.get_world_counter()
    cache = CacheView{TestResults}(:BasicTest, world)
    mi = method_instance(basic_node, (Int,); world, method_table)
    ci3 = get!(cache, mi) do
        create_ci(cache, mi)
    end
    res3 = results(cache, ci3)
    res3.result = my_compile(mi)
    @test res3.result == 60  # 30 * 2
    @test compile_count[] == 2  # incremented
end

@testset "multiple dispatch" begin
    method_table = @eval @MethodTable $(gensym(:method_table))

    function dispatch_node end
    add_method(method_table, dispatch_node, (Int,), 100)
    add_method(method_table, dispatch_node, (Float64,), 200)

    compile_count = Ref(0)
    function my_compile(mi)
        source = mi.def.source
        compile_count[] += 1
        source + 1
    end

    world = Base.get_world_counter()
    cache = CacheView{TestResults}(:DispatchTest, world)
    mi_int = method_instance(dispatch_node, (Int,); world, method_table)
    mi_float = method_instance(dispatch_node, (Float64,); world, method_table)

    # Different types → different cache entries, each compiles once
    ci_int = get!(cache, mi_int) do
        create_ci(cache, mi_int)
    end
    res_int = results(cache, ci_int)
    res_int.result = my_compile(mi_int)
    @test res_int.result == 101
    @test compile_count[] == 1

    ci_float = get!(cache, mi_float) do
        create_ci(cache, mi_float)
    end
    res_float = results(cache, ci_float)
    res_float.result = my_compile(mi_float)
    @test res_float.result == 201
    @test compile_count[] == 2

    # Cache hits - no recompilation
    ci_int2 = get!(cache, mi_int) do
        create_ci(cache, mi_int)
    end
    res_int2 = results(cache, ci_int2)
    @test res_int2.result == 101
    @test ci_int2 === ci_int
    @test res_int2 === res_int
    @test compile_count[] == 2  # unchanged

    ci_float2 = get!(cache, mi_float) do
        create_ci(cache, mi_float)
    end
    res_float2 = results(cache, ci_float2)
    @test res_float2.result == 201
    @test ci_float2 === ci_float
    @test res_float2 === res_float
    @test compile_count[] == 2  # unchanged

    # Redefine only Int method → only Int recompiles
    add_method(method_table, dispatch_node, (Int,), 50)
    world = Base.get_world_counter()
    cache = CacheView{TestResults}(:DispatchTest, world)
    mi_int = method_instance(dispatch_node, (Int,); world, method_table)
    ci_int3 = get!(cache, mi_int) do
        create_ci(cache, mi_int)
    end
    res_int3 = results(cache, ci_int3)
    res_int3.result = my_compile(mi_int)
    @test res_int3.result == 51
    @test compile_count[] == 3

    # Float64 still uses cached version (need to re-lookup mi after world change)
    mi_float = method_instance(dispatch_node, (Float64,); world, method_table)
    ci_float3 = get!(cache, mi_float) do
        create_ci(cache, mi_float)
    end
    res_float3 = results(cache, ci_float3)
    @test res_float3.result == 201
    @test compile_count[] == 3  # unchanged
end

@testset "missing method" begin
    method_table = @eval @MethodTable $(gensym(:method_table))

    function missing_node end
    # No method registered

    # Returns nothing when no method found
    world = Base.get_world_counter()
    mi = method_instance(missing_node, (Int,); world, method_table)
    @test mi === nothing
end

@testset "dependency invalidation" begin
    method_table = @eval @MethodTable $(gensym(:method_table))

    function parent_node end
    function child_node end
    add_method(method_table, child_node, (Int,), :child_ir)
    add_method(method_table, parent_node, (Int,), :parent_ir)

    child_compile_count = Ref(0)
    parent_compile_count = Ref(0)

    # Child compilation: creates CI with no deps
    function compile_child(cache, mi)
        child_compile_count[] += 1
        ci = get!(cache, mi) do
            create_ci(cache, mi)
        end
        res = results(cache, ci)
        res.result = :child_ir
    end

    # Parent compilation: creates CI with dependency on child
    function compile_parent(cache, mi)
        parent_compile_count[] += 1
        child_mi = method_instance(child_node, (Int,); world=cache.world, method_table)

        # Must compile child first to establish dependency
        compile_child(cache, child_mi)

        # Create CI with dependency
        ci = create_ci(cache, mi; deps=[child_mi])
        cache[mi] = ci
        res = results(cache, ci)
        res.result = :parent_ir
    end

    world = Base.get_world_counter()
    cache = CacheView{TestResults}(:DepTest, world)
    child_mi = method_instance(child_node, (Int,); world, method_table)
    parent_mi = method_instance(parent_node, (Int,); world, method_table)

    # Compile child first
    compile_child(cache, child_mi)
    @test child_compile_count[] == 1

    # Compile parent (depends on child)
    compile_parent(cache, parent_mi)
    @test parent_compile_count[] == 1
    @test child_compile_count[] == 2  # child recompiled during parent compilation

    # Cache hits
    ci_child = get!(cache, child_mi) do
        create_ci(cache, child_mi)
    end
    res_child = results(cache, ci_child)
    @test res_child.result === :child_ir
    ci_parent = get!(cache, parent_mi) do
        create_ci(cache, parent_mi)
    end
    res_parent = results(cache, ci_parent)
    @test res_parent.result === :parent_ir

    # Redefine child → child recompiles
    add_method(method_table, child_node, (Int,), :new_child_ir)
    world = Base.get_world_counter()
    cache = CacheView{TestResults}(:DepTest, world)
    child_mi = method_instance(child_node, (Int,); world, method_table)
    compile_child(cache, child_mi)
    @test child_compile_count[] == 3

    # Parent should also recompile due to dependency (cache miss)
    parent_mi = method_instance(parent_node, (Int,); world, method_table)
    compile_parent(cache, parent_mi)
    @test parent_compile_count[] == 2
end

@testset "method table isolation" begin
    method_table_a = @eval @MethodTable $(gensym(:method_table_a))
    method_table_b = @eval @MethodTable $(gensym(:method_table_b))

    function isolated_node end
    add_method(method_table_a, isolated_node, (Int,), :ir_a)
    add_method(method_table_b, isolated_node, (Int,), :ir_b)

    world = Base.get_world_counter()
    cache_a = CacheView{TestResults}(:IsolationA, world)
    cache_b = CacheView{TestResults}(:IsolationB, world)
    mi_a = method_instance(isolated_node, (Int,); world, method_table=method_table_a)
    mi_b = method_instance(isolated_node, (Int,); world, method_table=method_table_b)

    ci_a = get!(cache_a, mi_a) do
        create_ci(cache_a, mi_a)
    end
    res_a = results(cache_a, ci_a)
    res_a.result = mi_a.def.source
    ci_b = get!(cache_b, mi_b) do
        create_ci(cache_b, mi_b)
    end
    res_b = results(cache_b, ci_b)
    res_b.result = mi_b.def.source

    @test res_a.result === :ir_a
    @test res_b.result === :ir_b
end

end

#==============================================================================#
# Examples
#==============================================================================#

@testset "Examples" begin
    function find_sources(path::String, sources=String[])
        if isdir(path)
            for entry in readdir(path)
                find_sources(joinpath(path, entry), sources)
            end
        elseif endswith(path, ".jl")
            push!(sources, path)
        end
        sources
    end

    examples_dir = joinpath(@__DIR__, "..", "examples")
    examples = find_sources(examples_dir)
    filter!(file -> readline(file) != "# EXCLUDE FROM TESTING", examples)

    for example in examples
        name = splitext(basename(example))[1]
        @testset "$name" begin
            cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) $example`
            @test success(pipeline(cmd; stdout=devnull, stderr=devnull))
        end
    end
end

include("utils.jl")
include("precompile.jl")

end
