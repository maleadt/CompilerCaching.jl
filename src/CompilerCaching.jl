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
    SpecializedResult{V}

A specialized inference result for specific argument types.
"""
struct SpecializedResult{V}
    argtypes::Vector{Any}
    inner::V
    src::Any
    rettype::Any
    rettype_const::Any
end

"""
    CachedResult{V}

Mutable wrapper for analysis results that supports both generic and const-specialized
entries. Stored once in the CI's `analysis_results` chain at creation time. Const-prop
entries are accumulated by pushing to `const_entries`.
"""
mutable struct CachedResult{V}
    inner::V
    const_entries::Vector{SpecializedResult{V}}
    CachedResult{V}(inner::V) where V = new{V}(inner, SpecializedResult{V}[])
end

"""
    get_invoke_mi(stmt::Expr) -> Union{MethodInstance, Nothing}

Version-portable extraction of the callee MethodInstance from an `:invoke` statement.
On 1.12+ the first arg may be a CodeInstance; on 1.11 it's a MethodInstance directly.
"""
function get_invoke_mi(stmt::Expr)
    target = stmt.args[1]
    @static if VERSION >= v"1.12-"
        target isa Core.CodeInstance && return CC.get_ci_mi(target)
    end
    target isa Core.MethodInstance && return target
    return nothing
end

"""
    extract_invoke_argtypes(stmt::Expr, src::Core.CodeInfo, sptypes) -> Vector{Any}

Extract inferred argument types at each position of an `:invoke` call using `CC.argextype`.
Skips the invoke target at position 1.
"""
function extract_invoke_argtypes(stmt::Expr, src::Core.CodeInfo, sptypes)
    argtypes = Any[]
    for j in 2:length(stmt.args)
        if src.slottypes !== nothing
            push!(argtypes, CC.argextype(stmt.args[j], src, sptypes))
        else
            push!(argtypes, Any)
        end
    end
    return argtypes
end

"""
    extract_invoke_argtypes(stmt, src, sptypes, parent_argtypes) -> Vector{Any}

Like `extract_invoke_argtypes`, but resolves `Argument(i)` nodes using the parent's
const-enriched argtypes instead of the source's generic slot types.
"""
function extract_invoke_argtypes(stmt::Expr, src::Core.CodeInfo, sptypes,
                                 parent_argtypes::Vector{Any})
    argtypes = Any[]
    for j in 2:length(stmt.args)
        arg = stmt.args[j]
        if arg isa Core.Argument && checkbounds(Bool, parent_argtypes, arg.n)
            push!(argtypes, parent_argtypes[arg.n])
        elseif src.slottypes !== nothing
            push!(argtypes, CC.argextype(arg, src, sptypes))
        else
            push!(argtypes, Any)
        end
    end
    return argtypes
end

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
                V = $results_type(interp.$cache_field)
                $CC.stack_analysis_result!(caller.result, $CachedResult{V}(V()))
                @invoke $CC.finish!(interp::$CC.AbstractInterpreter, caller::$CC.InferenceState,
                                    validation_world::UInt, time_before::UInt64)
            end
        end
    else
        quote
            function $CC.finish!(interp::$InterpType, caller::$CC.InferenceState)
                V = $results_type(interp.$cache_field)
                $CC.stack_analysis_result!(caller.result, $CachedResult{V}(V()))
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

Retrieve the generic typed results struct from a CodeInstance's `analysis_results` chain.
Throws if no V is found - this indicates @setup_caching wasn't used correctly
or create_ci wasn't called.
"""
function results(::Type{V}, ci::Core.CodeInstance)::V where V
    cached = CC.traverse_analysis_results(ci) do @nospecialize result
        result isa CachedResult{V} ? result : nothing
    end
    @assert cached !== nothing "CodeInstance missing $V results - ensure @setup_caching is used or create_ci was called"
    return cached.inner
end

results(::CacheView{K,V}, ci::Core.CodeInstance) where {K,V} = results(V, ci)

"""
    results(::Type{V}, ci::CodeInstance, argtypes::Vector{Any})::V
    results(cache::CacheView{K,V}, ci::CodeInstance, argtypes::Vector{Any})::V

Retrieve const-specialized results for a specific set of argument types.
"""
function results(::Type{V}, ci::Core.CodeInstance, argtypes::Vector{Any})::V where V
    cached = CC.traverse_analysis_results(ci) do @nospecialize result
        result isa CachedResult{V} ? result : nothing
    end
    @assert cached !== nothing "CodeInstance missing $V results for argtypes $argtypes"
    for entry in cached.const_entries
        entry.argtypes == argtypes && return entry.inner
    end
    error("CodeInstance missing $V results for argtypes $argtypes")
end

results(::CacheView{K,V}, ci::Core.CodeInstance, argtypes::Vector{Any}) where {K,V} =
    results(V, ci, argtypes)

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
function Base.get(cache::CacheView, mi::Core.MethodInstance)
    ci = get(cache, mi, nothing)
    ci === nothing && throw(KeyError(mi))
    return ci
end
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
public captured_globals

"""
    captured_globals(source) -> iterable of GlobalRef

Return the `GlobalRef`s a foreign IR `source` captures. Override this for
your custom IR type to enable automatic binding-invalidation tracking; the
default returns none.

[`create_ci`](@ref) consults this hook to wire the referenced bindings into
the runtime's invalidation mechanism without the caller having to thread
them through manually.

Report only bindings the IR actually reads: over-reporting causes spurious
invalidations. The result must be stable for the lifetime of `source` —
edges are registered on every CI created for a method using this source.
"""
captured_globals(@nospecialize(source)) = ()

"""
    add_method(mt, f, arg_types, source) -> Method

Register a method with custom source IR in the cache's method table.

# Arguments
- `mt::Core.MethodTable` - The method table to add the method to
- `f::Function` - The function to add a method to
- `arg_types::Tuple` - Argument types for this method
- `source` - Custom IR to store (any type)

If `source` captures global bindings, override [`captured_globals`](@ref)
for its type. The bindings are then registered as edges of every
[`create_ci`](@ref) call for this method, so the cached code is
invalidated whenever any of them is replaced.

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
    m.nospecialize = UInt32(0)
    m.external_mt = mt
    m.slot_syms = ""
    m.source = source

    # For non-CodeInfo sources, mark the source as scanned *before* publishing
    # the method, so any concurrent retrieve_code_info / jl_scan_method_source_now
    # skips its CodeInfo-only scan (which would otherwise crash trying to
    # uncompress foreign IR). For CodeInfo sources the bit is left untouched so
    # Julia's own scan still runs.
    @static if VERSION >= v"1.12-"
        if !isa(source, Core.CodeInfo)
            @atomic m.did_scan_source |= 0x1
        end
    end

    ccall(:jl_method_table_insert, Cvoid, (Any, Any, Any), mt, m, nothing)

    return m
end


#==============================================================================#
# Method lookup
#==============================================================================#

export method_instance, match_method_instance

"""
    match_method_instance(f, tt; world, method_table) -> Union{MethodInstance, Nothing}

Look up the MethodInstance for function `f` with argument types `tt` using
method matching instead of cached dispatch lookup.

Unlike `method_instance`, this function accepts non-dispatch tuples (abstract
argument types) without crashing. Use this for compile-time analysis where
argument types may not be fully concrete.

Returns `nothing` if no unique matching method is found.
"""
function match_method_instance(@nospecialize(f), @nospecialize(tt);
                               world::UInt=Base.get_world_counter(),
                               method_table::Union{Core.MethodTable,Nothing}=nothing)
    sig = Base.signature_type(f, tt)
    matches = Base._methods_by_ftype(sig, method_table, 1, world)
    matches === nothing && return nothing
    length(matches) != 1 && return nothing
    if VERSION >= v"1.12-"
        return Base.specialize_method(matches[1]::Core.MethodMatch)
    else
        return CC.specialize_method(matches[1]::Core.MethodMatch)
    end
end

# jl_get_specialization1 doesn't support custom method tables (hardcodes jl_nothing).
# Reimplement its pipeline (match → normalize → specialize) with method table support.
function _specialization1(@nospecialize(sig), world::UInt, method_table::Core.MethodTable)
    matches = Base._methods_by_ftype(sig, method_table, 1, world)
    matches === nothing && return nothing
    length(matches) != 1 && return nothing
    match = matches[1]::Core.MethodMatch
    m = match.method
    ti = match.spec_types
    env = match.sparams
    @static if VERSION >= v"1.12-"
        tt = ccall(:jl_normalize_to_compilable_sig, Any, (Any, Any, Any, Cint),
                   ti, env, m, Cint(1))
    else # 1.11: extra jl_methtable_t* first param
        mt = ccall(:jl_method_get_table, Any, (Any,), m)
        tt = ccall(:jl_normalize_to_compilable_sig, Any, (Any, Any, Any, Any, Cint),
                   mt, ti, env, m, Cint(1))
    end
    tt === nothing && return nothing
    if tt !== ti
        pair = ccall(:jl_type_intersection_with_env, Any, (Any, Any),
                     tt, m.sig)::Core.SimpleVector
        env = pair[2]::Core.SimpleVector
    end
    return ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
                 (Any, Any, Any), m, tt, env)
end

# Before JuliaLang/julia#60718, `jl_method_lookup_by_tt` did not correctly cache overlay
# methods, causing lookups to fail or return stale global entries, so don't use the cache.
# Use jl_get_specialization1 instead, which uses jl_matching_methods (not cached dispatch)
# and returns compileable signatures (with proper vararg widening).
# Fixed in 1.14.0-DEV.1581, backported to 1.13.0-beta2, 1.12.5, and 1.11.9.
@static if (VERSION >= v"1.14.0-DEV.1581" ||
            v"1.13.0-beta2" <= VERSION < v"1.14-" ||
            v"1.12.5" <= VERSION < v"1.13-" ||
            v"1.11.9" <= VERSION < v"1.12-")
    @inline function method_instance(@nospecialize(f), @nospecialize(tt);
                                     world::UInt=Base.get_world_counter(),
                                     method_table::Union{Core.MethodTable,Nothing}=nothing)
        Base.method_instance(f, tt; world, method_table)
    end
elseif VERSION >= v"1.13-"
    # 3-arg jl_get_specialization1, returns jl_nothing on failure
    @inline function method_instance(@nospecialize(f), @nospecialize(tt);
                                     world::UInt=Base.get_world_counter(),
                                     method_table::Union{Core.MethodTable,Nothing}=nothing)
        sig = Base.signature_type(f, tt)
        @assert isdispatchtuple(sig)
        if method_table === nothing
            mi = ccall(:jl_get_specialization1, Any, (Any, Csize_t, Cint),
                       sig, world, Cint(0))
            return mi === nothing ? nothing : mi::Core.MethodInstance
        else
            return _specialization1(sig, world, method_table)
        end
    end
elseif VERSION >= v"1.12-"
    # 3-arg jl_get_specialization1, returns NULL on failure
    @inline function method_instance(@nospecialize(f), @nospecialize(tt);
                                     world::UInt=Base.get_world_counter(),
                                     method_table::Union{Core.MethodTable,Nothing}=nothing)
        sig = Base.signature_type(f, tt)
        @assert isdispatchtuple(sig)
        if method_table === nothing
            ptr = ccall(:jl_get_specialization1, Ptr{Cvoid}, (Any, Csize_t, Cint),
                        sig, world, Cint(0))
            return ptr == C_NULL ? nothing : unsafe_pointer_to_objref(ptr)::Core.MethodInstance
        else
            return _specialization1(sig, world, method_table)
        end
    end
else # 1.11: 5-arg jl_get_specialization1 (extra min_valid/max_valid out-params), returns NULL
    @inline function method_instance(@nospecialize(f), @nospecialize(tt);
                                     world::UInt=Base.get_world_counter(),
                                     method_table::Union{Core.MethodTable,Nothing}=nothing)
        sig = Base.signature_type(f, tt)
        @assert isdispatchtuple(sig)
        if method_table === nothing
            min_valid = Ref{Csize_t}(1)
            max_valid = Ref{Csize_t}(typemax(Csize_t))
            ptr = ccall(:jl_get_specialization1, Ptr{Cvoid},
                        (Any, Csize_t, Ref{Csize_t}, Ref{Csize_t}, Cint),
                        sig, world, min_valid, max_valid, Cint(0))
            return ptr == C_NULL ? nothing : unsafe_pointer_to_objref(ptr)::Core.MethodInstance
        else
            return _specialization1(sig, world, method_table)
        end
    end
end

"""
    method_instance(f, tt; world, method_table) -> Union{MethodInstance, Nothing}

Look up the compileable MethodInstance for function `f` with argument types `tt`.

Uses `jl_get_specialization1` (or `Base.method_instance` on Julia ≥ 1.14) to return
a compileable specialization with proper vararg widening.
Requires `tt` to be a dispatch tuple (fully concrete argument types).
Use [`match_method_instance`](@ref) for compile-time lookups where types
may not be fully resolved.

Returns `nothing` if no matching method is found.
"""
method_instance


#==============================================================================#
# Populating the cache
#==============================================================================#

export typeinf!, create_ci, get_source, get_codeinfos

"""
    typeinf!(cache, interp, mi) -> Nothing

Run type inference on `mi` and store the resulting CodeInstance in the cache.
Eagerly compiles all callees and stores their source so `get_codeinfos` works.
The CodeInstance can be retrieved with `get(cache, mi)`.
"""
function typeinf!(cache::CacheView, interp::CC.AbstractInterpreter,
                   mi::Core.MethodInstance)
    @static if VERSION >= v"1.12.0-DEV.1434"
        ci = CC.typeinf_ext(interp, mi, CC.SOURCE_MODE_NOT_REQUIRED)
        ci === nothing && return nothing

        # Eagerly compile all callees and store source
        has_compilequeue = VERSION >= v"1.13.0-DEV.499" || v"1.12-beta3" <= VERSION < v"1.13-"
        if has_compilequeue
            workqueue = CC.CompilationQueue(; interp)
            push!(workqueue, ci)
        else
            workqueue = Core.CodeInstance[ci]
            inspected = IdSet{Core.CodeInstance}()
        end

        while !isempty(workqueue)
            callee = pop!(workqueue)
            if has_compilequeue
                CC.isinspected(workqueue, callee) && continue
                CC.markinspected!(workqueue, callee)
            else
                callee in inspected && continue
                push!(inspected, callee)
            end

            # now make sure everything has source code, if desired
            callee_mi = CC.get_ci_mi(callee)
            if CC.use_const_api(callee)
                # const-return: get_source will synthesize CodeInfo, no need to store
                continue
            end

            src = CC.typeinf_code(interp, callee_mi, true)
            if src isa Core.CodeInfo
                # Store source so get_codeinfos can retrieve it later
                if (@atomic callee.inferred) === nothing
                    @atomic callee.inferred = src
                end
                if has_compilequeue
                    sptypes = CC.sptypes_from_meth_instance(callee_mi)
                    CC.collectinvokes!(workqueue, src, sptypes)
                else
                    CC.collectinvokes!(workqueue, src)
                end
            end
        end
    elseif VERSION >= v"1.12.0-DEV.15"
        inferred_ci = CC.typeinf_ext_toplevel(interp, mi, CC.SOURCE_MODE_FORCE_SOURCE)
        @assert inferred_ci !== nothing "Inference of $mi failed"

        # inference should have populated our cache
        ci = get(cache, mi)

        # if ci is rettype_const, the inference result won't have been cached
        # (because it is normally not supposed to be used ever again).
        # to avoid the need to re-infer, set that field here.
        if ci.inferred === nothing
            cache[mi] = inferred_ci
        end
    else
        # Julia 1.11: typeinf_ext_toplevel returns CodeInfo, not CI
        src = CC.typeinf_ext_toplevel(interp, mi)
        @assert src !== nothing "Inference of $mi failed"

        # inference should have populated our cache
        ci = get(cache, mi)

        # if ci is rettype_const, the inference result won't have been cached
        # (because it is normally not supposed to be used ever again).
        # to avoid the need to re-infer, set that field here.
        if ci.inferred === nothing
            @atomic ci.inferred = src
        end
    end
    return
end

"""
    typeinf!(cache, interp, mi, argtypes) -> Nothing

Run const-seeded type inference on `mi` with enriched `argtypes` and store the result
as a `CachedResult` entry on the generic CI's `analysis_results` chain.

Uses Julia's ephemeral `:local` inference mode (same as internal const-prop) so no
new CodeInstance is created. The const-specialized source and return type are stored
alongside the generic result for later retrieval via `results(cache, ci, argtypes)`
and `get_source(ci, argtypes)`.
"""
function typeinf!(cache::CacheView{K,V}, interp::CC.AbstractInterpreter,
                  mi::Core.MethodInstance, argtypes::Vector{Any}) where {K,V}
    # Ensure generic CI exists
    ci = get(cache, mi, nothing)
    if ci === nothing
        typeinf!(cache, interp, mi)
        ci = get(cache, mi, nothing)
        ci === nothing && return nothing
    end

    # Find the CachedResult on this CI
    cached = CC.traverse_analysis_results(ci) do @nospecialize result
        result isa CachedResult{V} ? result : nothing
    end
    @assert cached !== nothing "CodeInstance missing CachedResult{$V}"

    # Check if we already have a const-prop result for these argtypes
    for entry in cached.const_entries
        entry.argtypes == argtypes && return
    end

    # Compute overridden_by_const
    𝕃 = CC.typeinf_lattice(interp)
    @static if VERSION >= v"1.12-"
        default_argtypes = CC.matching_cache_argtypes(𝕃, mi)
        overridden = BitVector(undef, length(argtypes))
        for i in eachindex(argtypes)
            overridden[i] = !CC.is_lattice_equal(𝕃, argtypes[i], default_argtypes[i])
        end
    else
        # Pack varargs: on 1.11 matching_cache_argtypes packs trailing
        # args into a Tuple (returning nargs elements), but invoke stmts
        # list them individually, so we must pack argtypes to match.
        argtypes = CC.va_process_argtypes(𝕃, argtypes, mi)
        default_argtypes, _ = CC.matching_cache_argtypes(𝕃, mi)
        overridden = CC.BitVector(undef, length(argtypes))
        for i in eachindex(argtypes)
            CC.setindex!(overridden, !CC.is_lattice_equal(𝕃, argtypes[i], default_argtypes[i]), i)
        end
    end

    # Run ephemeral inference (:local mode, no result.ci)
    inf_result = CC.InferenceResult(mi, argtypes, overridden)
    frame = CC.InferenceState(inf_result, #=cache_mode=# :local, interp)
    if frame === nothing
        return nothing
    end
    CC.typeinf(interp, frame)

    # Convert OptimizationState → CodeInfo (preserves :invoke stmts)
    src = inf_result.src
    if src isa CC.OptimizationState
        src = CC.ir_to_codeinf!(src)
    end

    # Extract V from ephemeral InferenceResult's analysis_results (stacked by finish!)
    v = CC.traverse_analysis_results(inf_result) do @nospecialize r
        r isa CachedResult{V} ? r.inner : nothing
    end
    if v === nothing
        v = V()
    end

    # Compute rettype_const
    rettype = inf_result.result
    rettype_const = rettype isa CC.Const ? rettype.val : nothing

    # Store const-prop entry on the mutable CachedResult
    # (must happen before recursive walk so the duplicate check on lines 441-443 prevents cycles)
    entry = SpecializedResult{V}(argtypes, v, src, rettype, rettype_const)
    push!(cached.const_entries, entry)

    # Recursively const-seed callees with propagated const argtypes.
    # Walk the *generic* source (which has :invoke stmts pointing to callee CIs)
    # to discover callees — the const-optimized source has :invoke stmts too, but
    # the generic source gives us stable callee CIs for cache lookups.
    generic_src = get_source(ci)
    if generic_src isa Core.CodeInfo
        sptypes = CC.sptypes_from_meth_instance(mi)
        for stmt in generic_src.code
            if stmt isa Expr && stmt.head === :(=)
                stmt = stmt.args[2]
            end
            if stmt isa Expr && (stmt.head === :invoke ||
                    (VERSION >= v"1.12-" && stmt.head === :invoke_modify))
                callee_mi = get_invoke_mi(stmt)
                callee_mi === nothing && continue
                callee_argtypes = extract_invoke_argtypes(stmt, generic_src, sptypes, argtypes)
                typeinf!(cache, interp, callee_mi, callee_argtypes)
            end
        end
    end

    return
end

"""
    create_ci(cache::CacheView{K,V}, mi; deps) -> CodeInstance

Create a CodeInstance for `mi` with proper owner, typed results, and backedges.

Creates a new CodeInstance with:
- Owner set to `cache.owner`
- A fresh `V()` instance in analysis_results
- Backedges registered for all dependencies in `deps`
- Per-CI binding edges, so that the resulting CodeInstance is invalidated
  whenever any binding the source captures is replaced. The set of
  `GlobalRef`s is taken from [`captured_globals(mi.def.source)`](@ref captured_globals).

Used for foreign mode where inference doesn't run. The CI participates in
Julia's invalidation mechanism via backedges registered from `deps` (callee
methods) and the `captured_globals` hook (referenced global bindings).

The asymmetry between `deps` (explicit kwarg) and bindings (implicit trait)
is intentional. Captured bindings are a property of the source IR — fixed at
method definition and shared across every specialization — so it's natural
to pin them to the source type once via [`captured_globals`](@ref) and have
`create_ci` consult them. Dependencies, by contrast, are discovered per
compilation: the same method may invoke different callees depending on the
argument types of `mi`.
"""
function create_ci(cache::CacheView{K,V}, mi::Core.MethodInstance;
                   deps::Vector{Core.MethodInstance}=Core.MethodInstance[]) where {K,V}
    owner = cache.owner

    @static if VERSION >= v"1.12-"
        binding_edges = Core.Binding[]
        if isa(mi.def, Core.Method) && isdefined(mi.def, :source)
            for e in captured_globals(mi.def.source)
                push!(binding_edges,
                      e isa Core.Binding ? e : convert(Core.Binding, e::GlobalRef))
            end
        end
        edges = isempty(deps) && isempty(binding_edges) ?
            Core.svec() : Core.svec(deps..., binding_edges...)
    else
        # Julia 1.11 has no per-CI edges field
        edges = isempty(deps) ? Core.svec() : Core.svec(deps...)
    end

    # Create typed results instance via CachedResult{V}
    ar = CC.AnalysisResults(CachedResult{V}(V()), CC.NULL_ANALYSIS_RESULTS)

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

    @static if VERSION >= v"1.12-"
        # Register the CI as a direct edge of each captured binding. We
        # deliberately bypass `jl_maybe_add_binding_backedge` (which would
        # register the *Method* and route same-module invalidations through
        # `invalidate_method_for_globalref!`); that path tries to
        # `_uncompressed_ir(method)` and crashes on non-CodeInfo source.
        # Going CI-direct means binding replacement invalidates the CI via
        # the `isa(edge, CodeInstance)` branch in `invalidate_code_for_globalref!`.
        for b in binding_edges
            ccall(:jl_add_binding_backedge, Cvoid, (Any, Any), b, ci)
        end
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

"""
    get_source(ci::CodeInstance) -> Union{CodeInfo, Nothing}

Retrieve CodeInfo from a CodeInstance's inferred field.
Handles decompression if stored as String, and generates synthetic
CodeInfo for const-return functions.

Returns `nothing` if CodeInfo cannot be retrieved (e.g., for runtime
functions inferred by NativeInterpreter that don't store source).
For the root CI from `typeinf!`, this should always return valid CodeInfo.
"""
function get_source(ci::Core.CodeInstance)
    mi = @static if VERSION >= v"1.12-"
        CC.get_ci_mi(ci)
    else
        ci.def::Core.MethodInstance
    end

    src = @atomic :monotonic ci.inferred
    if src === nothing
        # For const-return functions, generate synthetic CodeInfo
        if CC.use_const_api(ci)
            @static if VERSION >= v"1.13.0-DEV.1121"
                src = CC.codeinfo_for_const(CC.NativeInterpreter(), mi,
                    CC.WorldRange(ci.min_world, ci.max_world),
                    ci.edges, ci.rettype_const)
            elseif VERSION >= v"1.12-"
                src = CC.codeinfo_for_const(CC.NativeInterpreter(), mi, ci.rettype_const)
                # Work around 1.12/1.13 not setting nargs/isva in `codeinfo_for_const`
                @static if v"1.12-" <= VERSION < v"1.14.0-DEV.60"
                    if src.nargs == 0 && mi.def isa Method
                        src.nargs = mi.def.nargs
                        src.isva = mi.def.isva
                    end
                end
            end
        end
    elseif src isa String
        # Decompress if stored as compressed String
        src = ccall(:jl_uncompress_ir, Ref{Core.CodeInfo},
                    (Any, Any, Any), mi.def::Method, ci, src)
    end
    return src isa Core.CodeInfo ? src : nothing
end

"""
    get_source(ci::CodeInstance, argtypes::Vector{Any}) -> Union{CodeInfo, Nothing}

Retrieve const-specialized CodeInfo from a CodeInstance's `CachedResult` chain.
Returns `nothing` if no const-prop entry exists for the given argtypes.
"""
function get_source(ci::Core.CodeInstance, argtypes::Vector{Any})
    cached = CC.traverse_analysis_results(ci) do @nospecialize result
        result isa CachedResult ? result : nothing
    end
    cached === nothing && return nothing
    for entry in cached.const_entries
        if entry.argtypes == argtypes
            src = entry.src
            return src isa Core.CodeInfo ? src : nothing
        end
    end
    return nothing
end

"""
    get_codeinfos(ci::CodeInstance) -> Vector{Pair{CodeInstance, CodeInfo}}

Collect CodeInstance/CodeInfo pairs by walking forward edges from a root CI.

On Julia 1.12+, walks `:invoke` statements to collect callees transitively.
On Julia 1.11, returns only the root entry.

Requires that `typeinf!` was called first to populate source for all callees.
"""
function get_codeinfos(ci::Core.CodeInstance)
    codeinfos = Pair{Core.CodeInstance, Core.CodeInfo}[]
    @static if VERSION >= v"1.12-"
        visited = IdSet{Core.CodeInstance}()
        workqueue = Core.CodeInstance[ci]
        while !isempty(workqueue)
            callee_ci = pop!(workqueue)
            callee_ci in visited && continue
            push!(visited, callee_ci)

            src = get_source(callee_ci)
            @assert src !== nothing "CodeInstance for $(CC.get_ci_mi(callee_ci)) has no source - ensure typeinf! was called"
            push!(codeinfos, callee_ci => src)

            for stmt in src.code
                if stmt isa Expr && stmt.head === :(=)
                    stmt = stmt.args[2]
                end
                if stmt isa Expr && (stmt.head === :invoke || stmt.head === :invoke_modify)
                    callee = stmt.args[1]
                    if callee isa Core.CodeInstance
                        push!(workqueue, callee)
                    end
                end
            end
        end
    else
        src = get_source(ci)
        src !== nothing && push!(codeinfos, ci => src)
    end
    return codeinfos
end

"""
    get_codeinfos(ci::CodeInstance, argtypes::Vector{Any}) -> Vector{Pair{CodeInstance, CodeInfo}}

Collect CodeInstance/CodeInfo pairs using the const-optimized source for the root CI
and generic source for all callees.

Delegates to `get_codeinfos(ci)` for the full callee walk, then swaps the root entry's
source with the const-optimized version from `get_source(ci, argtypes)`. Extra callees
from the generic walk are harmless (compiled but uncalled), and missing callees get
runtime dispatch stubs from `jl_emit_native`.

Falls back to `get_codeinfos(ci)` if no const entry exists for the given argtypes.
"""
function get_codeinfos(ci::Core.CodeInstance, argtypes::Vector{Any})
    const_src = get_source(ci, argtypes)
    if const_src === nothing
        return get_codeinfos(ci)
    end
    codeinfos = get_codeinfos(ci)
    # Swap root entry's source with const-optimized version
    idx = findfirst(p -> p.first === ci, codeinfos)
    if idx !== nothing
        codeinfos[idx] = ci => const_src
    else
        pushfirst!(codeinfos, ci => const_src)
    end
    return codeinfos
end

end # module CompilerCaching
