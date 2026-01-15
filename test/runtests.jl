using CompilerCaching
using Test
using Base.Experimental: @MethodTable

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
    my_compile(ctx) = begin
        source = ctx.mi.def.source
        compile_count[] += 1
        source * 2
    end

    # First call: cache miss, compile_fn invoked
    r1 = cached_compilation(my_compile, cache, basic_node, (Int,))
    @test something(r1) == 20  # 10 * 2
    @test compile_count[] == 1

    # Second call: cache hit, compile_fn NOT invoked
    r2 = cached_compilation(my_compile, cache, basic_node, (Int,))
    @test something(r2) == 20
    @test compile_count[] == 1  # still 1

    # Redefine method → invalidates cache, recompile
    add_method(cache, basic_node, (Int,), 30)
    r3 = cached_compilation(my_compile, cache, basic_node, (Int,))
    @test something(r3) == 60  # 30 * 2
    @test compile_count[] == 2  # incremented
end

@testset "multiple dispatch" begin
    method_table = @eval @MethodTable $(gensym(:method_table))
    cache = CompilerCache(:DispatchTest, method_table)

    function dispatch_node end
    add_method(cache, dispatch_node, (Int,), 100)
    add_method(cache, dispatch_node, (Float64,), 200)

    compile_count = Ref(0)
    my_compile(ctx) = begin
        source = ctx.mi.def.source
        compile_count[] += 1
        source + 1
    end

    # Different types → different cache entries, each compiles once
    r_int = cached_compilation(my_compile, cache, dispatch_node, (Int,))
    @test something(r_int) == 101
    @test compile_count[] == 1

    r_float = cached_compilation(my_compile, cache, dispatch_node, (Float64,))
    @test something(r_float) == 201
    @test compile_count[] == 2

    # Cache hits - no recompilation
    r_int2 = cached_compilation(my_compile, cache, dispatch_node, (Int,))
    @test something(r_int2) == 101
    @test compile_count[] == 2  # unchanged

    r_float2 = cached_compilation(my_compile, cache, dispatch_node, (Float64,))
    @test something(r_float2) == 201
    @test compile_count[] == 2  # unchanged

    # Redefine only Int method → only Int recompiles
    add_method(cache, dispatch_node, (Int,), 50)
    r_int3 = cached_compilation(my_compile, cache, dispatch_node, (Int,))
    @test something(r_int3) == 51
    @test compile_count[] == 3

    # Float64 still uses cached version
    r_float3 = cached_compilation(my_compile, cache, dispatch_node, (Float64,))
    @test something(r_float3) == 201
    @test compile_count[] == 3  # unchanged
end

@testset "nothing on missing method" begin
    method_table = @eval @MethodTable $(gensym(:method_table))
    cache = CompilerCache(:MissingTest, method_table)

    function missing_node end
    # No method registered

    my_compile(ctx) = ctx.mi.def.source

    # Returns nothing when no method found
    result = cached_compilation(my_compile, cache, missing_node, (Int,))
    @test result === nothing

    # Also test method_instance directly
    mi = method_instance(missing_node, (Int,); method_table=cache.mt)
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
    my_compile(ctx) = begin
        compile_count[] += 1
        ctx.mi.def.source
    end

    # First key combination
    keys1 = (opt_level=1, debug=false)
    r1 = cached_compilation(my_compile, cache, sharded_node, (Int,), keys1)
    @test something(r1) == 42
    @test compile_count[] == 1

    # Same key combination → cache hit
    r2 = cached_compilation(my_compile, cache, sharded_node, (Int,), keys1)
    @test something(r2) == 42
    @test compile_count[] == 1  # unchanged

    # Different key combination → cache miss (different shard)
    keys2 = (opt_level=2, debug=false)
    r3 = cached_compilation(my_compile, cache, sharded_node, (Int,), keys2)
    @test something(r3) == 42
    @test compile_count[] == 2  # new compilation

    # Yet another key combination
    keys3 = (opt_level=1, debug=true)
    r4 = cached_compilation(my_compile, cache, sharded_node, (Int,), keys3)
    @test something(r4) == 42
    @test compile_count[] == 3  # another new compilation

    # Back to first key combination → still cached
    r5 = cached_compilation(my_compile, cache, sharded_node, (Int,), keys1)
    @test something(r5) == 42
    @test compile_count[] == 3  # unchanged
end

@testset "CompilationContext provides mi" begin
    method_table = @eval @MethodTable $(gensym(:method_table))
    cache = CompilerCache(:CtxTest, method_table)

    function ctx_node end
    add_method(cache, ctx_node, (Int,), :my_source)

    captured_mi = Ref{Any}(nothing)
    captured_source = Ref{Any}(nothing)

    my_compile(ctx) = begin
        captured_mi[] = ctx.mi
        captured_source[] = ctx.mi.def.source
        :compiled
    end

    cached_compilation(my_compile, cache, ctx_node, (Int,))

    @test captured_source[] === :my_source
    @test captured_mi[] isa Core.MethodInstance
    @test captured_mi[].def.name === :ctx_node
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

    child_compile(ctx) = begin
        child_compile_count[] += 1
        :child_compiled
    end

    parent_compile(ctx) = begin
        parent_compile_count[] += 1
        # Register dependency on child
        child_mi = method_instance(child_node, (Int,); method_table=cache.mt)
        register_dependency!(ctx, child_mi)
        :parent_compiled
    end

    # Compile child first
    cached_compilation(child_compile, cache, child_node, (Int,))
    @test child_compile_count[] == 1

    # Compile parent (depends on child)
    cached_compilation(parent_compile, cache, parent_node, (Int,))
    @test parent_compile_count[] == 1

    # Cache hits
    cached_compilation(child_compile, cache, child_node, (Int,))
    cached_compilation(parent_compile, cache, parent_node, (Int,))
    @test child_compile_count[] == 1
    @test parent_compile_count[] == 1

    # Redefine child → child recompiles
    add_method(cache, child_node, (Int,), :new_child_ir)
    cached_compilation(child_compile, cache, child_node, (Int,))
    @test child_compile_count[] == 2

    # Parent should also recompile due to dependency
    cached_compilation(parent_compile, cache, parent_node, (Int,))
    @test parent_compile_count[] >= 1
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

    compile_a(ctx) = (result_a[] = ctx.mi.def.source; ctx.mi.def.source)
    compile_b(ctx) = (result_b[] = ctx.mi.def.source; ctx.mi.def.source)

    cached_compilation(compile_a, cache_a, isolated_node, (Int,))
    cached_compilation(compile_b, cache_b, isolated_node, (Int,))

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
    my_compile(ctx) = begin
        source = ctx.mi.def.source
        captured_ir[] = source
        length(source.nodes)
    end

    result = cached_compilation(my_compile, cache, complex_node, (Int,))
    @test something(result) == 3
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
    my_compile(ctx) = begin
        compile_count[] += 1
        # Source is compressed Julia source (can use Base.uncompressed_ast if needed)
        # Return something based on the method
        ctx.mi.def.name
    end

    result = cached_compilation(my_compile, cache, overlay_double, (Int,))
    # Overlay methods may have gensym'd names like "#overlay_double"
    @test occursin("overlay_double", string(something(result)))
    @test compile_count[] == 1

    # Cache hit
    result2 = cached_compilation(my_compile, cache, overlay_double, (Int,))
    @test occursin("overlay_double", string(something(result2)))
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
    my_compile(ctx) = begin
        compile_count[] += 1
        # For global methods, source may be compressed (String/Vector{UInt8})
        # Use Base.uncompressed_ast() if you need CodeInfo
        # Here we just return the method name
        ctx.mi.def.name
    end

    result = cached_compilation(my_compile, cache, global_test_fn, (Int,))
    @test something(result) === :global_test_fn
    @test compile_count[] == 1

    # Cache hit
    result2 = cached_compilation(my_compile, cache, global_test_fn, (Int,))
    @test something(result2) === :global_test_fn
    @test compile_count[] == 1  # unchanged
end

@testset "global MT with sharding keys" begin
    # Global MT with sharding keys (e.g., for GPUCompiler-style usage)
    ShardKeys = @NamedTuple{opt_level::Int}
    cache = CompilerCache{ShardKeys}(:GlobalShardTest)

    global_sharded_fn(x::Float64) = x * 2.0

    compile_count = Ref(0)
    my_compile(ctx) = begin
        compile_count[] += 1
        ctx.mi.def.name
    end

    # Different sharding keys = different cache entries
    r1 = cached_compilation(my_compile, cache, global_sharded_fn, (Float64,), (opt_level=1,))
    @test something(r1) === :global_sharded_fn
    @test compile_count[] == 1

    r2 = cached_compilation(my_compile, cache, global_sharded_fn, (Float64,), (opt_level=2,))
    @test something(r2) === :global_sharded_fn
    @test compile_count[] == 2  # different shard

    r3 = cached_compilation(my_compile, cache, global_sharded_fn, (Float64,), (opt_level=1,))
    @test something(r3) === :global_sharded_fn
    @test compile_count[] == 2  # cache hit
end

end # Mode 3

end # @testset "ForeignCompiler"

println("All tests passed!")
