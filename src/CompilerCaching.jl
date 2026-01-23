# CompilerCaching.jl - Reusable package for compiler result caching
#
# Leverages Julia's Method/MethodInstance/CodeInstance infrastructure to provide:
# - Lazy compilation with caching
# - Type-based specialization and dispatch
# - Automatic invalidation when methods are redefined
# - Transitive dependency tracking
#
# Requires Julia 1.11+

module CompilerCaching

using Base.Experimental: @MethodTable
const CC = Core.Compiler

include("utils.jl")


#==============================================================================#
# CacheView structure
#==============================================================================#

export CacheView, @setup_caching, results

"""
    CacheView{K, V}

A cache into a cache partition at a specific world age. Serves as the main entry point
for cached compilation.
"""
struct CacheView{K, V}
    owner::K
    world::UInt
    CacheView{K,V}(owner, world::UInt) where {K,V} = new{K,V}(convert(K, owner), world)
end

CacheView{V}(owner::K, world::UInt) where {K,V} = CacheView{K,V}(owner, world)

"""
    @setup_caching InterpreterType.cache_field

Generate the required methods for an AbstractInterpreter to work with CompilerCaching.

The cache field must be a `CacheView{K, V}` where `V` is your typed results struct.
The macro generates:
- `CC.cache_owner(interp)` returning the cache's owner token
- `CC.finish!(interp, caller, ...)` that stacks a new `V()` instance in analysis results
"""
macro setup_caching(expr)
    # Parse InterpreterType.cache_field
    if !(expr isa Expr && expr.head == :.)
        error("Expected InterpreterType.cache_field, e.g., @setup_caching MyInterpreter.cache")
    end
    InterpType = expr.args[1]
    cache_field = expr.args[2]
    if cache_field isa QuoteNode
        cache_field = cache_field.value
    end

    finish_method = if hasmethod(CC.finish!, Tuple{CC.AbstractInterpreter, CC.InferenceState, UInt, UInt64})
        quote
            function $CC.finish!(interp::$InterpType, caller::$CC.InferenceState,
                                 validation_world::UInt, time_before::UInt64)
                $CC.stack_analysis_result!(caller.result, $results_type(interp.$cache_field)())
                @invoke $CC.finish!(interp::$CC.AbstractInterpreter, caller::$CC.InferenceState,
                                    validation_world::UInt, time_before::UInt64)
            end
        end
    else
        quote
            function $CC.finish!(interp::$InterpType, caller::$CC.InferenceState)
                $CC.stack_analysis_result!(caller.result, $results_type(interp.$cache_field)())
                @invoke $CC.finish!(interp::$CC.AbstractInterpreter, caller::$CC.InferenceState)
            end
        end
    end

    quote
        $CC.cache_owner(interp::$InterpType) = $cache_owner(interp.$cache_field)
        $finish_method
    end |> esc
end

"""
    cache_owner(cache::CacheView)

Returns the owner token for use as CodeInstance.owner.
"""
cache_owner(cache::CacheView) = cache.owner

"""
    results_type(cache::CacheView{K,V}) -> Type{V}

Returns the results type V for a cache view.
"""
results_type(::CacheView{K,V}) where {K,V} = V

"""
    results(::Type{V}, ci::CodeInstance)::V
    results(cache::CacheView{K,V}, ci::CodeInstance)::V

Retrieve the typed results struct from a CodeInstance's `analysis_results` chain.
Throws if no V is found - this indicates @setup_caching wasn't used correctly
or create_ci wasn't called.
"""
function results(::Type{V}, ci::Core.CodeInstance)::V where V
    result = CC.traverse_analysis_results(ci) do @nospecialize result
        result isa V ? result : nothing
    end
    @assert result !== nothing "CodeInstance missing $V results - ensure @setup_caching is used or create_ci was called"
    return result
end

results(::CacheView{K,V}, ci::Core.CodeInstance) where {K,V} = results(V, ci)

@static if VERSION >= v"1.14-"
    function code_cache(cache::CacheView)
        world_range = CC.WorldRange(cache.world)
        return CC.InternalCodeCache(cache_owner(cache), world_range)
    end
else
    function code_cache(cache::CacheView)
        cc = CC.InternalCodeCache(cache_owner(cache))
        return CC.WorldView(cc, cache.world)
    end
end

# Expose InternalCodeCache functionality
Base.haskey(cache::CacheView, mi::Core.MethodInstance) = CC.haskey(code_cache(cache), mi)
Base.get(cache::CacheView, mi::Core.MethodInstance, default) = CC.get(code_cache(cache), mi, default)
Base.getindex(cache::CacheView, mi::Core.MethodInstance) = CC.getindex(code_cache(cache), mi)
Base.setindex!(cache::CacheView, ci::Core.CodeInstance, mi::Core.MethodInstance) = CC.setindex!(code_cache(cache), ci, mi)


#==============================================================================#
# Cache access
#==============================================================================#

"""
    Base.get!(f::Function, cache::CacheView, mi::MethodInstance) -> CodeInstance

Get an existing CodeInstance or create one using `f()`.

Standard dict interface: returns existing CI if found, otherwise calls `f()`
which must return a CodeInstance, stores it, and returns it.

# Example (foreign mode)
```julia
ci = get!(cache, mi) do
    create_ci(cache, mi; deps)
end
```
"""
function Base.get!(f::Function, cache::CacheView, mi::Core.MethodInstance)
    ci = get(cache, mi, nothing)
    ci !== nothing && return ci
    ci = f()::Core.CodeInstance
    cache[mi] = ci
    return ci
end



#==============================================================================#
# Foreign method registration
#==============================================================================#

export add_method

"""
    add_method(mt, f, arg_types, source) -> Method

Register a method with custom source IR in the cache's method table.

# Arguments
- `mt::Core.MethodTable` - The method table to add the method to
- `f::Function` - The function to add a method to
- `arg_types::Tuple` - Argument types for this method
- `source` - Custom IR to store (any type)

# Returns
The created `Method` object.
"""
function add_method(mt::Core.MethodTable, f::Function, arg_types::Tuple, source)
    sig = Tuple{typeof(f), arg_types...}

    m = ccall(:jl_new_method_uninit, Any, (Any,), parentmodule(f))

    m.name = nameof(f)
    m.module = parentmodule(f)
    m.file = Symbol("foreign")
    m.line = Int32(0)
    m.sig = sig
    m.nargs = Int32(1 + length(arg_types))
    m.isva = false
    m.called = UInt32(0)
    m.nospecialize = UInt32(0)
    m.external_mt = mt
    m.slot_syms = ""
    m.source = source

    ccall(:jl_method_table_insert, Cvoid, (Any, Any, Any), mt, m, nothing)

    return m
end


#==============================================================================#
# Method lookup
#==============================================================================#

export method_instance

# Before JuliaLang/julia#60718, `jl_method_lookup_by_tt` did not correctly cache overlay
# methods, causing lookups to fail or return stale global entries, so don't use the cache.
@static if VERSION >= v"1.14.0-DEV.1581"
    # NOTE: is being backported
    using Base: method_instance
else
    function method_instance(@nospecialize(f), @nospecialize(tt);
                             world::UInt=Base.get_world_counter(),
                             method_table::Union{Core.MethodTable,Nothing}=nothing)
        sig = Base.signature_type(f, tt)
        match, _ = CC._findsup(sig, method_table, world)
        match === nothing && return nothing
        CC.specialize_method(match)::Core.MethodInstance
    end
end

"""
    method_instance(f, tt; world, method_table) -> Union{MethodInstance, Nothing}

Look up the MethodInstance for function `f` with argument types `tt`.

Uses Julia's cached method lookup (`jl_method_lookup_by_tt`) for fast lookups.
Returns `nothing` if no matching method is found.
"""
method_instance


#==============================================================================#
# Populating the cache
#==============================================================================#

export typeinf!, create_ci

"""
    typeinf!(cache, interp, mi) -> Vector{Pair{CodeInstance, CodeInfo}}

Run type inference on `mi` and store the resulting CodeInstances in the cache.

Uses the provided interpreter, which must implement the `CC.AbstractInterpreter`
interface. The resulting CodeInstances are stored in the cache for later retrieval.

Returns a vector of (CodeInstance, CodeInfo) pairs:
- **Julia 1.12+**: `[ci => CodeInfo, ...]` - root + callees in topological order
- **Julia 1.11**: `[ci => CodeInfo]` - only root entry

The root CI is always the first entry: `first(result)`.

# Consumer Patterns

For native codegen (uses full vector on 1.12+, callback on 1.11):
```julia
codeinfos = typeinf!(cache, interp, mi)
julia_codegen(cache, mi, codeinfos)  # handles version differences internally
```

For IR transformation (uses first entry, inflates to IRCode):
```julia
codeinfos = typeinf!(cache, interp, mi)
ci, src = first(codeinfos)
ir = CC.inflate_ir(src, CC.get_ci_mi(ci))
```
"""
function typeinf!(cache::CacheView, interp::CC.AbstractInterpreter,
                   mi::Core.MethodInstance)
    @static if VERSION >= v"1.12.0-DEV.1434"
        # Modern API: returns CodeInstance, use SOURCE_MODE to control caching
        # (SOURCE_MODE_FORCE_SOURCE was renamed to SOURCE_MODE_GET_SOURCE)
        source_mode = @static if isdefined(CC, :SOURCE_MODE_GET_SOURCE)
            CC.SOURCE_MODE_GET_SOURCE
        else
            CC.SOURCE_MODE_FORCE_SOURCE
        end
        ci = CC.typeinf_ext(interp, mi, source_mode)
        @assert ci !== nothing "Inference of $mi failed"

        # Collect all code that needs compilation (including callees)
        codeinfos = Pair{Core.CodeInstance, Core.CodeInfo}[]

        @static if VERSION >= v"1.13.0-DEV.499" || v"1.12-beta3" <= VERSION < v"1.13-"
            workqueue = CC.CompilationQueue(; interp)
            push!(workqueue, ci)
            while !isempty(workqueue)
                callee = pop!(workqueue)
                CC.isinspected(workqueue, callee) && continue
                CC.markinspected!(workqueue, callee)

                callee_mi = CC.get_ci_mi(callee)
                if CC.use_const_api(callee)
                    @static if VERSION >= v"1.13.0-DEV.1121"
                        src = CC.codeinfo_for_const(interp, callee_mi,
                            CC.WorldRange(callee.min_world, callee.max_world),
                            callee.edges, callee.rettype_const)
                    else
                        src = CC.codeinfo_for_const(interp, callee_mi, callee.rettype_const)
                        # Work around 1.12/1.13 not setting nargs/isva in `codeinfo_for_const`
                        # This is fixed by JuliaLang/julia#59413, and will be backported.
                        @static if v"1.12-" <= VERSION < v"1.14.0-DEV.60"
                            if src.nargs == 0 && callee_mi.def isa Method
                                src.nargs = callee_mi.def.nargs
                                src.isva = callee_mi.def.isva
                            end
                        end
                    end
                else
                    src = CC.typeinf_code(interp, callee_mi, true)
                end
                if src isa Core.CodeInfo
                    sptypes = CC.sptypes_from_meth_instance(callee_mi)
                    CC.collectinvokes!(workqueue, src, sptypes)
                    push!(codeinfos, callee => src)
                end
            end
        else
            # Older 1.12 API
            workqueue = Core.CodeInstance[ci]
            inspected = IdSet{Core.CodeInstance}()
            while !isempty(workqueue)
                callee = pop!(workqueue)
                callee in inspected && continue
                push!(inspected, callee)

                callee_mi = CC.get_ci_mi(callee)
                if CC.use_const_api(callee)
                    src = CC.codeinfo_for_const(interp, callee_mi, callee.rettype_const)
                    # Work around 1.12/1.13 not setting nargs/isva in `codeinfo_for_const`
                    # This is fixed by JuliaLang/julia#59413, and will be backported.
                    @static if v"1.12-" <= VERSION < v"1.14.0-DEV.60"
                        if src.nargs == 0 && callee_mi.def isa Method
                            src.nargs = callee_mi.def.nargs
                            src.isva = callee_mi.def.isva
                        end
                    end
                else
                    src = CC.typeinf_code(interp, callee_mi, true)
                end
                if src isa Core.CodeInfo
                    CC.collectinvokes!(workqueue, src)
                    push!(codeinfos, callee => src)
                end
            end
        end

        return codeinfos
    elseif VERSION >= v"1.12.0-DEV.15"
        # Julia 1.12 early API
        source_mode = @static if isdefined(CC, :SOURCE_MODE_GET_SOURCE)
            CC.SOURCE_MODE_GET_SOURCE
        else
            CC.SOURCE_MODE_FORCE_SOURCE
        end
        ci = CC.typeinf_ext_toplevel(interp, mi, source_mode)
        @assert ci !== nothing "Inference of $mi failed"
        return Pair{Core.CodeInstance, Core.CodeInfo}[]
    else
        # Julia 1.11 API - cache populated implicitly
        src = CC.typeinf_ext_toplevel(interp, mi)

        # Handle const-return case where ci.inferred may be nothing
        ci = get(cache, mi, nothing)
        @assert ci !== nothing "Inference of $mi failed"
        if ci.inferred === nothing
            @atomic ci.inferred = src
        end

        return Pair{Core.CodeInstance, Core.CodeInfo}[ci => src]
    end
end

"""
    create_ci(cache::CacheView{K,V}, mi; deps) -> CodeInstance

Create a CodeInstance for `mi` with proper owner, typed results, and backedges.

Creates a new CodeInstance with:
- Owner set to `cache.owner`
- A fresh `V()` instance in analysis_results
- Backedges registered for all dependencies in `deps`

Used for foreign mode where inference doesn't run. The CI participates in
Julia's invalidation mechanism via backedges registered from `deps`.
"""
function create_ci(cache::CacheView{K,V}, mi::Core.MethodInstance;
                   deps::Vector{Core.MethodInstance}=Core.MethodInstance[]) where {K,V}
    owner = cache.owner
    edges = isempty(deps) ? Core.svec() : Core.svec(deps...)

    # Create typed results instance via V()
    ar = CC.AnalysisResults(V(), CC.NULL_ANALYSIS_RESULTS)

    @static if VERSION >= v"1.12-"
        ci = Core.CodeInstance(mi, owner, Any, Any, nothing, nothing,
            Int32(0), cache.world, typemax(UInt), UInt32(0), ar, nothing, edges)
    else
        ci = Core.CodeInstance(mi, owner, Any, Any, nothing, nothing,
            Int32(0), cache.world, typemax(UInt), UInt32(0), UInt32(0), ar, UInt8(0))
    end

    # Register backedges for automatic invalidation
    if !isempty(deps)
        store_backedges(mi, ci, deps)
    end

    return ci
end

"""
    store_backedges(mi::MethodInstance, ci::CodeInstance, deps::Vector{MethodInstance})

Register backedges so Julia automatically invalidates cached code when dependencies change.
This enables Julia's built-in invalidation mechanism - when any dependency MI is
invalidated, the caller MI's CodeInstances will have their max_world reduced.
"""
function store_backedges(mi::Core.MethodInstance, ci::Core.CodeInstance,
                         deps::Vector{Core.MethodInstance})
    isa(mi.def, Method) || return  # don't add backedges to toplevel

    for dep_mi in deps
        @static if VERSION >= v"1.12-"
            # Julia 1.12+: pass CodeInstance as caller
            ccall(:jl_method_instance_add_backedge, Cvoid,
                  (Any, Any, Any), dep_mi, nothing, ci)
        else
            # Julia 1.11: pass MethodInstance as caller
            ccall(:jl_method_instance_add_backedge, Cvoid,
                  (Any, Any, Any), dep_mi, nothing, mi)
        end
    end
    nothing
end

end # module CompilerCaching
