# Foreign IR example - bypasses Julia's inference/compiler stack entirely
# - Methods registered with add_method storing custom IR
# - Compiler function transforms the foreign IR
# - Demonstrates: caching, redefinition, multiple dispatch

using CompilerCaching: CompilerCache, add_method, method_instance, cache!, cached_compilation, cached_inference

using Base: get_world_counter
using Base.Experimental: @MethodTable

@MethodTable FOREIGN_MT

const FOREIGN_CACHE = CompilerCache(:ForeignExample, FOREIGN_MT)


## foreign IR definition

struct ForeignIR
    op::Symbol
    value::Any
    calls::Vector{Tuple{Any, Tuple}}  # [(callee_func, arg_types), ...]

    ForeignIR(op::Symbol, value::Any) = new(op, value, Tuple{Any, Tuple}[])
    ForeignIR(op::Symbol, value::Any, calls::Vector) = new(op, value, calls)
end


## three-phase compilation

const compilations = Ref(0)

# Infer phase: handles dependency tracking via cache!
function infer(cache::CompilerCache, mi::Core.MethodInstance, world::UInt)
    ir = mi.def.source::ForeignIR
    deps = Core.MethodInstance[]

    # Recursively infer callees (no codegen/link) and collect dependencies
    for (callee_func, callee_tt) in ir.calls
        callee_mi = method_instance(callee_func, callee_tt; world, cache.method_table)
        callee_mi === nothing && error("No method for $callee_func with $callee_tt")

        # Only run inference for dependency - establishes CI and backedges
        cached_inference(cache, callee_mi, world; infer)
        push!(deps, callee_mi)
    end

    # Create CI with backedges for dependency tracking
    ci = cache!(cache, mi; world, deps)
    return [ci => ir]
end

# Codegen phase: "compile" by evaluating the operation
function codegen(cache::CompilerCache, mi::Core.MethodInstance, world::UInt, codeinfos)
    compilations[] += 1
    _, ir = only(codeinfos)

    if ir.op == :identity
        return ir.value
    elseif ir.op == :double
        return ir.value * 2
    elseif ir.op == :square
        return ir.value * ir.value
    else
        error("Unknown operation: $(ir.op)")
    end
end

# Link phase: just pass through the result
function link(cache::CompilerCache, mi::Core.MethodInstance, world::UInt, result)
    result
end


## high-level API

function call(f, args...)
    tt = Tuple{map(Core.Typeof, args)...}
    world = get_world_counter()
    mi = method_instance(f, tt; world, method_table=FOREIGN_CACHE.method_table)
    mi === nothing && throw(MethodError(f, args))

    cached_compilation(FOREIGN_CACHE, mi, world; infer, codegen, link)
end


## demo

# Define a function with foreign IR
function myop end
add_method(FOREIGN_CACHE, myop, (Int,), ForeignIR(:double, 21))

# First call compiles
result = call(myop, 0)  # argument value unused, IR has the value
@assert result == 42
@assert compilations[] == 1

# Second call uses cache
result = call(myop, 0)
@assert result == 42
@assert compilations[] == 1

# Redefine with different IR - invalidates cache
add_method(FOREIGN_CACHE, myop, (Int,), ForeignIR(:square, 7))
result = call(myop, 0)
@assert result == 49
@assert compilations[] == 2

# Repeated call still cached
result = call(myop, 0)
@assert result == 49
@assert compilations[] == 2

# Add method for different argument type - doesn't invalidate existing
add_method(FOREIGN_CACHE, myop, (Float64,), ForeignIR(:identity, 3.14))
result = call(myop, 0)  # Int version still cached
@assert result == 49
@assert compilations[] == 2

# Call the Float64 version - triggers new compilation
result = call(myop, 0.0)
@assert result == 3.14
@assert compilations[] == 3

# Both versions now cached
result = call(myop, 0)
@assert result == 49
result = call(myop, 0.0)
@assert result == 3.14
@assert compilations[] == 3

println("Basic assertions passed!")


## Transitive dependency example
#
# grandchild_node <- child_node <- parent_node
#
# When grandchild is redefined, both child and parent should be invalidated.

println("\n--- Transitive Dependency Demo ---")
compilations[] = 0

function grandchild_node end
function child_node end
function parent_node end

# grandchild: base function
add_method(FOREIGN_CACHE, grandchild_node, (Int,), ForeignIR(:identity, 1))

# child: calls grandchild
add_method(FOREIGN_CACHE, child_node, (Int,),
           ForeignIR(:double, 2, [(grandchild_node, (Int,))]))

# parent: calls child (which transitively depends on grandchild)
add_method(FOREIGN_CACHE, parent_node, (Int,),
           ForeignIR(:square, 3, [(child_node, (Int,))]))

# Compile all three
result = call(grandchild_node, 0)
@assert result == 1
@assert compilations[] == 1
println("grandchild compiled: $result (compilations: $(compilations[]))")

result = call(child_node, 0)
@assert result == 4  # 2 * 2
@assert compilations[] == 2  # child compiles, grandchild cached
println("child compiled: $result (compilations: $(compilations[]))")

result = call(parent_node, 0)
@assert result == 9  # 3 * 3
@assert compilations[] == 3  # parent compiles, child+grandchild cached
println("parent compiled: $result (compilations: $(compilations[]))")

# Cache hits - no recompilation
call(grandchild_node, 0)
call(child_node, 0)
call(parent_node, 0)
@assert compilations[] == 3
println("All cached (compilations still: $(compilations[]))")

# Now redefine grandchild
println("\nRedefining grandchild...")
add_method(FOREIGN_CACHE, grandchild_node, (Int,), ForeignIR(:identity, 100))

# grandchild should recompile
result = call(grandchild_node, 0)
@assert result == 100
@assert compilations[] == 4
println("grandchild recompiled: $result (compilations: $(compilations[]))")

# child should recompile due to dependency on grandchild
result = call(child_node, 0)
@assert result == 4  # still 2*2, the value in child's IR
@assert compilations[] == 5
println("child recompiled: $result (compilations: $(compilations[]))")

# parent should recompile due to transitive dependency via child
result = call(parent_node, 0)
@assert result == 9  # still 3*3, the value in parent's IR
@assert compilations[] == 6
println("parent recompiled: $result (compilations: $(compilations[]))")

# All cached again
call(grandchild_node, 0)
call(child_node, 0)
call(parent_node, 0)
@assert compilations[] == 6
println("All cached again (compilations still: $(compilations[]))")

println("\nAll transitive dependency assertions passed!")
