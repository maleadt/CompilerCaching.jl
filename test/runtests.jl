using CompilerCaching
using Test
using Base.Experimental: @MethodTable

# Test helper: wraps simple compile functions in get! API
function simple_cached_compilation(compile_fn, cache::CacheView, mi)
    get!(cache, mi, :result) do cache, mi
        result = compile_fn(mi)
        cache[mi] = CompilerCaching.create_ci(cache, mi)
        result
    end
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
    cache = CacheView(:GlobalTest, world)
    mi = method_instance(global_test_fn, (Int,); world)

    result = simple_cached_compilation(my_compile, cache, mi)
    @test result === :global_test_fn
    @test compile_count[] == 1

    # Cache hit
    result2 = simple_cached_compilation(my_compile, cache, mi)
    @test result2 === :global_test_fn
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
    ShardKeys = @NamedTuple{opt_level::Int}
    cache1 = CacheView{ShardKeys}(:GlobalShardTest, world, (opt_level=1,))
    cache2 = CacheView{ShardKeys}(:GlobalShardTest, world, (opt_level=2,))
    mi = method_instance(global_sharded_fn, (Float64,); world)

    # Different sharding keys = different cache entries
    r1 = simple_cached_compilation(my_compile, cache1, mi)
    @test r1 === :global_sharded_fn
    @test compile_count[] == 1

    r2 = simple_cached_compilation(my_compile, cache2, mi)
    @test r2 === :global_sharded_fn
    @test compile_count[] == 2  # different shard

    r3 = simple_cached_compilation(my_compile, cache1, mi)
    @test r3 === :global_sharded_fn
    @test compile_count[] == 2  # cache hit
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
    cache = CacheView(:OverlayTest, world)
    mi = method_instance(overlay_double, (Int,); world, method_table)

    result = simple_cached_compilation(my_compile, cache, mi)
    # Overlay methods may have gensym'd names like "#overlay_double"
    @test occursin("overlay_double", string(result))
    @test compile_count[] == 1

    # Cache hit
    result2 = simple_cached_compilation(my_compile, cache, mi)
    @test occursin("overlay_double", string(result2))
    @test compile_count[] == 1  # unchanged
end

@testset "inference integration" begin
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
    cache = CacheView(:InferenceTest, world)
    mi = method_instance(test_fn, (Int,); world)

    interp = TestInterpreter(cache)
    result = typeinf!(cache, interp, mi)

    # typeinf! now always returns Vector{Pair{CI, IR}}
    @test result isa Vector
    @test length(result) >= 1
    ci, ir = first(result)
    @test ci isa Core.CodeInstance
    @test ir isa Core.CodeInfo

    # Test const-return functions get CompilationResult wrapper
    # These functions return a constant and skip optimization, but finish! should still be called
    const_return_fn(x::Int) = nothing  # Returns constant `nothing`
    world2 = Base.get_world_counter()
    cache2 = CacheView(:InferenceTest, world2)
    mi2 = method_instance(const_return_fn, (Int,); world=world2)

    interp2 = TestInterpreter(cache2)
    result2 = typeinf!(cache2, interp2, mi2)

    @test length(result2) >= 1
    ci2, _ = first(result2)
    @test ci2 isa Core.CodeInstance
    # Verify it's actually a const-return CI (skip under coverage as it disables const-return)
    @test Core.Compiler.use_const_api(ci2) skip=(Base.JLOptions().code_coverage > 0)
    # The key test: finish! hook should have stacked our dict even for const-return
    @test CompilerCaching.cached_results(ci2) isa Dict{Symbol,Any}
end

@testset "compilation hook" begin
    calls = []

    # Define function and get world counter after definition
    test_func_hook(x::Int) = x + 1
    world = Base.get_world_counter()
    cache = CacheView(:HookTest, world)
    mi = method_instance(test_func_hook, (Int,); world)
    @test mi !== nothing

    # Test 1: Hook called on cache miss
    compile_hook!() do c, m
        push!(calls, :called)
    end

    get!(cache, mi, :result) do cache, mi
        cache[mi] = CompilerCaching.create_ci(cache, mi)
        :result
    end
    @test length(calls) == 1

    # Test 2: Hook called even on cache hit
    get!(cache, mi, :result) do cache, mi
        cache[mi] = CompilerCaching.create_ci(cache, mi)
        :result
    end
    @test length(calls) == 2  # Called again even though cached

    # Test 3: No hook when disabled
    compile_hook!(nothing)
    # Use different key to trigger miss
    get!(cache, mi, :other) do cache, mi
        :other_result
    end
    @test length(calls) == 2  # No new call

    # Test 4: Getter returns current hook
    f = (c, m) -> nothing
    compile_hook!(f)
    @test compile_hook() === f
    compile_hook!(nothing)
    @test compile_hook() === nothing
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
    cache = CacheView(:BasicTest, world)
    mi = method_instance(basic_node, (Int,); world, method_table)

    # First call: cache miss, compile_fn invoked
    r1 = simple_cached_compilation(my_compile, cache, mi)
    @test r1 == 20  # 10 * 2
    @test compile_count[] == 1

    # Second call: cache hit, compile_fn NOT invoked
    r2 = simple_cached_compilation(my_compile, cache, mi)
    @test r2 == 20
    @test compile_count[] == 1  # still 1

    # Redefine method → invalidates cache, recompile
    add_method(method_table, basic_node, (Int,), 30)
    world = Base.get_world_counter()
    cache = CacheView(:BasicTest, world)
    mi = method_instance(basic_node, (Int,); world, method_table)
    r3 = simple_cached_compilation(my_compile, cache, mi)
    @test r3 == 60  # 30 * 2
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
    cache = CacheView(:DispatchTest, world)
    mi_int = method_instance(dispatch_node, (Int,); world, method_table)
    mi_float = method_instance(dispatch_node, (Float64,); world, method_table)

    # Different types → different cache entries, each compiles once
    r_int = simple_cached_compilation(my_compile, cache, mi_int)
    @test r_int == 101
    @test compile_count[] == 1

    r_float = simple_cached_compilation(my_compile, cache, mi_float)
    @test r_float == 201
    @test compile_count[] == 2

    # Cache hits - no recompilation
    r_int2 = simple_cached_compilation(my_compile, cache, mi_int)
    @test r_int2 == 101
    @test compile_count[] == 2  # unchanged

    r_float2 = simple_cached_compilation(my_compile, cache, mi_float)
    @test r_float2 == 201
    @test compile_count[] == 2  # unchanged

    # Redefine only Int method → only Int recompiles
    add_method(method_table, dispatch_node, (Int,), 50)
    world = Base.get_world_counter()
    cache = CacheView(:DispatchTest, world)
    mi_int = method_instance(dispatch_node, (Int,); world, method_table)
    r_int3 = simple_cached_compilation(my_compile, cache, mi_int)
    @test r_int3 == 51
    @test compile_count[] == 3

    # Float64 still uses cached version (need to re-lookup mi after world change)
    mi_float = method_instance(dispatch_node, (Float64,); world, method_table)
    r_float3 = simple_cached_compilation(my_compile, cache, mi_float)
    @test r_float3 == 201
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

    # Child emit_ir: creates CI with no deps
    function child_emit_ir(c, m)
        child_compile_count[] += 1
        c[m] = create_ci(c, m)
        :child_ir
    end

    # Parent emit_ir: creates CI with dependency on child
    function parent_emit_ir(c, m)
        parent_compile_count[] += 1
        child_mi = method_instance(child_node, (Int,); world=c.world, method_table)
        c[m] = create_ci(c, m; deps=[child_mi])
        :parent_ir
    end

    world = Base.get_world_counter()
    cache = CacheView(:DepTest, world)
    child_mi = method_instance(child_node, (Int,); world, method_table)
    parent_mi = method_instance(parent_node, (Int,); world, method_table)

    # Compile child first
    get!(child_emit_ir, cache, child_mi, :result)
    @test child_compile_count[] == 1

    # Compile parent (depends on child)
    get!(parent_emit_ir, cache, parent_mi, :result)
    @test parent_compile_count[] == 1

    # Cache hits
    get!(child_emit_ir, cache, child_mi, :result)
    get!(parent_emit_ir, cache, parent_mi, :result)
    @test child_compile_count[] == 1
    @test parent_compile_count[] == 1

    # Redefine child → child recompiles
    add_method(method_table, child_node, (Int,), :new_child_ir)
    world = Base.get_world_counter()
    cache = CacheView(:DepTest, world)
    child_mi = method_instance(child_node, (Int,); world, method_table)
    get!(child_emit_ir, cache, child_mi, :result)
    @test child_compile_count[] == 2

    # Parent should also recompile due to dependency
    parent_mi = method_instance(parent_node, (Int,); world, method_table)
    get!(parent_emit_ir, cache, parent_mi, :result)
    @test parent_compile_count[] == 2
end

@testset "method table isolation" begin
    method_table_a = @eval @MethodTable $(gensym(:method_table_a))
    method_table_b = @eval @MethodTable $(gensym(:method_table_b))

    function isolated_node end
    add_method(method_table_a, isolated_node, (Int,), :ir_a)
    add_method(method_table_b, isolated_node, (Int,), :ir_b)

    result_a = Ref{Any}(nothing)
    result_b = Ref{Any}(nothing)

    world = Base.get_world_counter()
    cache_a = CacheView(:IsolationA, world)
    cache_b = CacheView(:IsolationB, world)
    mi_a = method_instance(isolated_node, (Int,); world, method_table=method_table_a)
    mi_b = method_instance(isolated_node, (Int,); world, method_table=method_table_b)

    function compile_a(m)
        result_a[] = m.def.source
        m.def.source
    end
    function compile_b(m)
        result_b[] = m.def.source
        m.def.source
    end

    simple_cached_compilation(compile_a, cache_a, mi_a)
    simple_cached_compilation(compile_b, cache_b, mi_b)

    @test result_a[] === :ir_a
    @test result_b[] === :ir_b
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
