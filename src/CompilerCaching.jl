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

"""
    @public foo, bar

Declare `foo, bar` as public API. Lowers to `public foo, bar` on 1.11+ (where `public`
is keyword syntax) and to a no-op on 1.10. Lets the rest of the module use a single
form regardless of Julia version, without `Meta.parse` workarounds.
"""
macro public(symbols_expr)
    syms = symbols_expr isa Symbol ? [symbols_expr] :
           symbols_expr.head === :tuple ? [a isa Symbol ? a : a.args[1] for a in symbols_expr.args] :
           [symbols_expr.args[1]]
    if VERSION >= v"1.11.0-DEV.469"
        esc(Expr(:public, syms...))
    else
        nothing
    end
end

@static if VERSION >= v"1.11"

using Base.Experimental: @MethodTable
const CC = Core.Compiler

include("utils.jl")


#==============================================================================#
# CacheView structure
#==============================================================================#

export CacheView, results, lookup

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

# Fast equality for argtypes vectors. Element-wise `==(::Any, ::Any)` does
# dynamic dispatch (~40 ns/element) — but `===` short-circuits to pointer
# equality, which is true for interned `Type` values and immutable
# `Core.Compiler.Const` instances with `===`-equal `.val`s. So most cache hits
# resolve in O(n) pointer compares; we only fall through to `==` on the rare
# element that isn't structurally identical.
@inline function argtypes_egal(a::Vector{Any}, b::Vector{Any})::Bool
    length(a) == length(b) || return false
    @inbounds for i in eachindex(a)
        a[i] === b[i] || a[i] == b[i] || return false
    end
    return true
end

"""
    CachedResult{V}

Mutable wrapper for analysis results that supports both generic and const-specialized
entries. Attached to a CI's `analysis_results` chain on first access (see
[`results`](@ref)) or at CI creation (see [`create_ci`](@ref)). Const-prop entries are
accumulated by pushing to `const_entries`.
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

# Owner choice and lookup cost

`owner` is stored on every cached `CodeInstance` as its `.owner` field. Cache
lookups (`get(cache, mi)`) ccall `jl_rettype_inferred`, which compares
`ci.owner` against the requested owner using `jl_egal` — i.e. structural
equality for immutable values, identity (`===`) for mutable ones.

The owner is passed to the ccall as `Any`, which forces the JIT to box it on
every call when its concrete type is not already a heap-allocated reference.
For hot paths this matters:

| owner type                            | per `get(cache, mi)` |
| ------------------------------------- | -------------------- |
| `Symbol`                              |   ~2.5 ns, 0 allocs  |
| `struct` (immutable, e.g. NamedTuple) |  ~10 ns,  small box  |
| `Tuple{Symbol, NamedTuple{…}}`        |  ~38 ns, ~112 B box  |

Prefer a `Symbol` if you don't need to shard. Otherwise, an immutable struct
or tuple is the right shape — it boxes on each lookup, but `CodeInstance`s
loaded from a package image still resolve because `jl_egal` matches by
content.

Avoid `mutable struct` owners: `jl_egal` falls back to identity for mutable
types, so a `CodeInstance` deserialized from a package image will not match
a fresh runtime instance even with identical fields.
"""
struct CacheView{K, V}
    owner::K
    world::UInt
    CacheView{K,V}(owner, world::UInt) where {K,V} = new{K,V}(convert(K, owner), world)
end

CacheView{V}(owner::K, world::UInt) where {K,V} = CacheView{K,V}(owner, world)

"""
    cache_owner(cache::CacheView)

Returns the owner token for use as CodeInstance.owner.
"""
cache_owner(cache::CacheView) = cache.owner

"""
    results_type(cache::CacheView{K,V}) -> Type{V}

The results type addressed by this cache view.
"""
results_type(::CacheView{K,V}) where {K,V} = V


## results attachment
#
# Results structs are attached to a CodeInstance's `analysis_results` chain *lazily*, on
# first access through `results` / `lookup`. This keeps inference entirely results-free:
# interpreters need no CompilerCaching-specific hooks (no `CC.finish!` override), and a
# single CI can carry results for multiple independent consumers (distinct `V` types
# coexist on the chain).

# `analysis_results` is declared const on Julia 1.11–1.13 (the C runtime mutates it in
# place after optimization), so plain `setfield!` is rejected and — more subtly — repeated
# `getfield`s may legally be CSE'd. We mutate through `jl_set_nth_field` (the same way
# `Serialization` writes const fields during deserialization) and re-read through an
# opaque ccall when re-checking under the lock.
const ANALYSIS_RESULTS_FIELD = something(findfirst(==(:analysis_results),
                                                   fieldnames(Core.CodeInstance)))

read_analysis_results(ci::Core.CodeInstance) =
    ccall(:jl_get_nth_field_checked, Any, (Any, Csize_t), ci, ANALYSIS_RESULTS_FIELD-1)

# Lock serializing chain mutations. Attachment is rare (once per (CI, V) pair), so a
# single global lock suffices. Plain (lock-free) reads are safe: chain nodes are
# immutable and only ever prepended, and `jl_set_nth_field` stores with release
# semantics.
#
# The C runtime also writes this field, without taking our lock: `jl_fill_codeinst`
# (and `jl_update_codeinst`) overwrite it wholesale when inference finishes. That's
# only safe because those writes happen while the CI is still private to the inference
# engine (jl_fill_codeinst asserts min_world == 1 / max_world == 0, i.e.
# pre-publication). Corollary: only attach results to CIs that have been published to
# the integrated cache — never to a CI still being inferred.
const attach_lock = ReentrantLock()

@noinline function attach_results!(::Type{V}, ci::Core.CodeInstance) where V
    Base.@lock attach_lock begin
        # re-read and re-check under the lock: another task may have attached while we
        # were acquiring it (the ccall also defeats CSE with the pre-lock traversal)
        chain = read_analysis_results(ci)
        head = chain isa CC.AnalysisResults ? chain : CC.NULL_ANALYSIS_RESULTS
        node = head
        while isdefined(node, :next)
            node.result isa CachedResult{V} && return node.result::CachedResult{V}
            node = node.next
        end

        cached = CachedResult{V}(V())
        ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), ci,
              ANALYSIS_RESULTS_FIELD-1, CC.AnalysisResults(cached, head))
        return cached
    end
end

function find_results(::Type{V}, ci::Core.CodeInstance) where V
    CC.traverse_analysis_results(ci) do @nospecialize result
        result isa CachedResult{V} ? result : nothing
    end
end

@inline function get_results(::Type{V}, ci::Core.CodeInstance) where V
    cached = find_results(V, ci)
    cached === nothing && (cached = attach_results!(V, ci))
    return cached::CachedResult{V}
end

"""
    results(::Type{V}, ci::CodeInstance)::V
    results(cache::CacheView{K,V}, ci::CodeInstance)::V

Retrieve the typed results struct of type `V` from a CodeInstance, creating and
attaching a fresh `V()` on first access. The same instance is returned for every
subsequent call with the same `V`, for the lifetime of the CodeInstance (including
across precompilation, when both the CI and its results are serialized into the
package image).

Mutations to a results struct attached to a CodeInstance that was loaded from a
*different* package image do not persist beyond the current (pre)compilation session;
only the image that serialized the CI owns its storage.
"""
results(::Type{V}, ci::Core.CodeInstance) where V = get_results(V, ci).inner

results(::CacheView{K,V}, ci::Core.CodeInstance) where {K,V} = results(V, ci)

"""
    results(::Type{V}, ci::CodeInstance, argtypes::Vector{Any})::V
    results(cache::CacheView{K,V}, ci::CodeInstance, argtypes::Vector{Any})::V

Retrieve const-specialized results for a specific set of argument types. Unlike the
generic accessor, const-specialized entries are only created by [`typeinf!`](@ref)
with `argtypes`; this throws if no matching entry exists.
"""
function results(::Type{V}, ci::Core.CodeInstance, argtypes::Vector{Any})::V where V
    cached = get_results(V, ci)
    for entry in cached.const_entries
        argtypes_egal(entry.argtypes, argtypes) && return entry.inner
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

"""
    lookup(cache::CacheView{K,V}, mi::MethodInstance) -> Union{Nothing, Tuple{CodeInstance, V}}
    lookup(cache::CacheView{K,V}, mi::MethodInstance, argtypes::Vector{Any}) ->
        Union{Nothing, Tuple{CodeInstance, V}}

Combined `get(cache, mi)` + `results(cache, ci[, argtypes])` accessor — single-pass
cache lookup. Returns `(ci, res)` when a `CodeInstance` is cached for `mi` (attaching
a fresh `V()` on first access), or `nothing` when there is no CI — or, with
`argtypes`, no matching const-prop entry.

Hot-path callers (e.g. `cufunction`) typically need both `ci` and `res` and walk
the same lookup multiple times across phases. Use `lookup` once and pass the
resulting `(ci, res)` pair down through phases instead of resolving them each
time.
"""
@inline function lookup(cache::CacheView{K,V}, mi::Core.MethodInstance) where {K,V}
    ci = get(cache, mi, nothing)
    ci === nothing && return nothing
    return (ci, results(V, ci))
end

@inline function lookup(cache::CacheView{K,V}, mi::Core.MethodInstance,
                        argtypes::Vector{Any}) where {K,V}
    ci = get(cache, mi, nothing)
    ci === nothing && return nothing
    cached = get_results(V, ci)
    for entry in cached.const_entries
        argtypes_egal(entry.argtypes, argtypes) && return (ci, entry.inner)
    end
    return nothing
end


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
@public captured_globals

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

# JuliaLang/julia#62001 specializes closed type-valued callees and arguments on
# `Core.TypeEgal` dispatch keys, making `Type{T}` elements non-dispatchable.
# `Base.signature_type` handles the callee (it takes a value), but not `Type{T}`
# spellings in a user-provided `tt`, so normalize those ourselves.
@static if isdefined(Core, :TypeEgal)
    @inline function dispatch_key(@nospecialize(t))
        if Base.isType(t)
            u = Base.type_parameter(t)
            Base.has_free_typevars(u) || return Core.TypeEgal{u}
        end
        return t
    end

    function signature_type(@nospecialize(f), @nospecialize(tt))
        sig = Base.signature_type(f, tt)
        u = Base.unwrap_unionall(sig)::DataType
        return Base.rewrap_unionall(Tuple{map(dispatch_key, u.parameters)...}, sig)
    end
else
    const signature_type = Base.signature_type
end

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
    sig = signature_type(f, tt)
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
function _specialization1(@nospecialize(sig), world::UInt,
                          method_table::Union{Core.MethodTable,Nothing})
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
# The overlay lookup issue is fixed in 1.13.0-beta2, 1.12.5, and 1.11.9. On
# 1.14+, `Base.method_instance` returns the dispatch-cache MI, and the compiler
# normalizes that to a compilable MI later. Do the same here, since this API is
# documented to return the compilation target.
@static if (v"1.13.0-beta2" <= VERSION < v"1.14-" ||
            v"1.12.5" <= VERSION < v"1.13-" ||
            v"1.11.9" <= VERSION < v"1.12-")
    @inline function method_instance(@nospecialize(f), @nospecialize(tt);
                                     world::UInt=Base.get_world_counter(),
                                     method_table::Union{Core.MethodTable,Nothing}=nothing)
        Base.method_instance(f, tt; world, method_table)
    end
elseif VERSION >= v"1.14-"
    @inline function method_instance(@nospecialize(f), @nospecialize(tt);
                                     world::UInt=Base.get_world_counter(),
                                     method_table::Union{Core.MethodTable,Nothing}=nothing)
        sig = signature_type(f, tt)
        @assert isdispatchtuple(sig)
        mi = Base.method_instance(sig; world, method_table)
        mi === nothing && return nothing
        return ccall(:jl_normalize_to_compilable_mi, Any, (Any,), mi)::Core.MethodInstance
    end
elseif VERSION >= v"1.13-"
    # 3-arg jl_get_specialization1, returns jl_nothing on failure
    @inline function method_instance(@nospecialize(f), @nospecialize(tt);
                                     world::UInt=Base.get_world_counter(),
                                     method_table::Union{Core.MethodTable,Nothing}=nothing)
        sig = signature_type(f, tt)
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
        sig = signature_type(f, tt)
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
        sig = signature_type(f, tt)
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

# Clear the interpreter-local const-prop cache: a `Vector{InferenceResult}` (1.12)
# or `Compiler.InferenceCache` (1.13+, which lacks `empty!`).
function reset_inference_cache!(interp::CC.AbstractInterpreter)
    cache = CC.get_inference_cache(interp)
    @static if isdefined(Core.Compiler, :InferenceCache)
        if cache isa CC.InferenceCache
            empty!(cache.results)
            empty!(cache.index)
            return
        end
    end
    empty!(cache)
    return
end

"""
    typeinf!(interp, mi) -> Union{CodeInstance, Nothing}

Run type inference on `mi` using `interp`, storing the resulting `CodeInstance`
in Julia's integrated cache (partitioned by `CC.cache_owner(interp)`). Eagerly
compiles all callees and stores their source so [`get_codeinfos`](@ref) works.

Returns the root `CodeInstance` (or `nothing` if inference failed). Subsequent
calls for the same `mi` and world are no-ops — the existing CI is returned.

The eager callee walk only follows `:invoke` edges that refer to a `CodeInstance`.
Optimized source (in particular source reused from the cache) can also contain
`:invoke` statements targeting a bare `MethodInstance`; those callees are not
compiled here, but are resolved lazily by [`get_codeinfos(interp, ci)`](@ref).
"""
function typeinf!(interp::CC.AbstractInterpreter, mi::Core.MethodInstance)
    @static if VERSION >= v"1.12.0-DEV.1434"
        # `mi` is inferred exactly as requested; callers wanting a compileable
        # specialization should normalize before using it as a cache key.
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

            # Reuse source already stored on the CI (by inference, or by a previous
            # walk) instead of unconditionally calling `typeinf_code`, which re-runs
            # inference and optimization without consulting the cache. This makes
            # repeated walks over an already-populated graph (e.g. `cached_results`
            # followed by the back-end's compile) traversal-only.
            src = get_source(callee)
            if src === nothing
                # Standalone re-inference must behave like a fresh `jl_typeinf` entry:
                # tombstoned (`LimitedAccuracy`) const-prop results left by the deeper
                # root cascade would otherwise suppress const-prop that succeeds in
                # this shallower context (JuliaGPU/CUDA.jl#3185).
                reset_inference_cache!(interp)
                src = CC.typeinf_code(interp, callee_mi, true)
                # Store source so get_codeinfos can retrieve it later
                if src isa Core.CodeInfo && (@atomic callee.inferred) === nothing
                    @atomic callee.inferred = src
                end
            end
            if src isa Core.CodeInfo
                if has_compilequeue
                    sptypes = CC.sptypes_from_meth_instance(callee_mi)
                    CC.collectinvokes!(workqueue, src, sptypes)
                else
                    CC.collectinvokes!(workqueue, src)
                end
            end
        end
        return ci
    elseif VERSION >= v"1.12.0-DEV.15"
        cache = CacheView{Nothing}(CC.cache_owner(interp), CC.get_inference_world(interp))
        inferred_ci = CC.typeinf_ext_toplevel(interp, mi, CC.SOURCE_MODE_FORCE_SOURCE)
        @assert inferred_ci !== nothing "Inference of $mi failed"

        # inference should have populated the cache
        ci = get(cache, mi, nothing)
        ci === nothing && return nothing

        # if ci is rettype_const, the inference result won't have been cached
        # (because it is normally not supposed to be used ever again).
        # to avoid the need to re-infer, set that field here.
        if ci.inferred === nothing
            cache[mi] = inferred_ci
        end
        return ci
    else
        # Julia 1.11: typeinf_ext_toplevel returns CodeInfo, not CI
        cache = CacheView{Nothing}(CC.cache_owner(interp), CC.get_inference_world(interp))
        src = CC.typeinf_ext_toplevel(interp, mi)
        @assert src !== nothing "Inference of $mi failed"

        # inference should have populated the cache
        ci = get(cache, mi, nothing)
        ci === nothing && return nothing

        # if ci is rettype_const, the inference result won't have been cached.
        # to avoid the need to re-infer, set that field here.
        if ci.inferred === nothing
            @atomic ci.inferred = src
        end
        return ci
    end
end

"""
    typeinf!(cache::CacheView{K,V}, interp, mi, argtypes) -> Nothing

Run const-seeded type inference on `mi` with enriched `argtypes` and store the result
as a `SpecializedResult{V}` entry on the generic CI's `CachedResult{V}`.

Uses Julia's ephemeral `:local` inference mode (same as internal const-prop) so no
new CodeInstance is created. The const-specialized source and return type are stored
alongside the generic result for later retrieval via `results(cache, ci, argtypes)`
and `get_source(ci, argtypes)`.

Unlike the generic `typeinf!(interp, mi)`, this form takes a `CacheView`: const-prop
entries are stored typed, so the results type `V` must be known up front. The cache
view must match the interpreter's owner and world.
"""
function typeinf!(cache::CacheView{K,V}, interp::CC.AbstractInterpreter,
                  mi::Core.MethodInstance, argtypes::Vector{Any}) where {K,V}
    @assert cache.owner === CC.cache_owner(interp) "CacheView owner does not match interpreter"
    @assert cache.world == CC.get_inference_world(interp) "CacheView world does not match interpreter"

    # Ensure generic CI exists
    ci = get(cache, mi, nothing)
    if ci === nothing
        typeinf!(interp, mi)
        ci = get(cache, mi, nothing)
        ci === nothing && return nothing
    end

    cached = get_results(V, ci)

    # Check if we already have a const-prop result for these argtypes
    for entry in cached.const_entries
        argtypes_egal(entry.argtypes, argtypes) && return
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

    v = V()

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
- Backedges registered for all dependencies in `deps`. A dependency may be a
  `MethodInstance`, meaning any compilation of that method, or a
  `CodeInstance`, meaning that exact compilation on Julia 1.12 and newer.
  Julia 1.11 has no per-CodeInstance forward-edge field, so there a
  `CodeInstance` dependency degrades to its `MethodInstance`.
- Per-CI binding edges, so that the resulting CodeInstance is invalidated
  whenever any binding the source captures is replaced. The set of
  `GlobalRef`s is taken from [`captured_globals(mi.def.source)`](@ref captured_globals).

Used for foreign mode where inference doesn't run.

The asymmetry between `deps` (explicit kwarg) and bindings (implicit trait)
is intentional. Captured bindings are a property of the source IR — fixed at
method definition and shared across every specialization — so it's natural
to pin them to the source type once via [`captured_globals`](@ref) and have
`create_ci` consult them. Dependencies, by contrast, are discovered per
compilation: the same method may invoke different callees depending on the
argument types of `mi`.
"""
const CompilationDependency = Union{Core.MethodInstance,Core.CodeInstance}
@public CompilationDependency

dependency_mi(mi::Core.MethodInstance) = mi
dependency_mi(ci::Core.CodeInstance) = @static if VERSION >= v"1.12-"
    CC.get_ci_mi(ci)
else
    ci.def::Core.MethodInstance
end

function create_ci(cache::CacheView{K,V}, mi::Core.MethodInstance;
                   deps::AbstractVector=CompilationDependency[]) where {K,V}
    owner = cache.owner
    world = cache.world

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
            Int32(0), world, typemax(UInt), UInt32(0), ar, nothing, edges)
    else
        ci = Core.CodeInstance(mi, owner, Any, Any, nothing, nothing,
            Int32(0), world, typemax(UInt), UInt32(0), UInt32(0), ar, UInt8(0))
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
    store_backedges(mi::MethodInstance, ci::CodeInstance, deps)

Register backedges so Julia automatically invalidates cached code when dependencies
change. On Julia 1.12 and newer, the caller is the new `CodeInstance` and its
forward edges preserve whether each dependency is an entire `MethodInstance` or
one exact `CodeInstance`. On Julia 1.11, both kinds degrade to a
`MethodInstance`-to-`MethodInstance` backedge.
"""
function store_backedges(mi::Core.MethodInstance, ci::Core.CodeInstance,
                         deps::AbstractVector)
    isa(mi.def, Method) || return  # don't add backedges to toplevel

    for dep in deps
        dep_mi = dependency_mi(dep)
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
        if argtypes_egal(entry.argtypes, argtypes)
            src = entry.src
            return src isa Core.CodeInfo ? src : nothing
        end
    end
    return nothing
end

"""
    get_codeinfos(interp::AbstractInterpreter, ci::CodeInstance) ->
        Vector{Pair{CodeInstance, CodeInfo}}

Collect the `CodeInstance`/`CodeInfo` pairs needed to generate code for `ci`, walking
`:invoke` edges transitively from the root. On Julia 1.12+ the result is closed under
direct-call edges, making it suitable for closed-world code generation
(`jl_emit_native`, which cannot look up missing callees during codegen).

`interp` is used to repair gaps that cache history can leave behind:

- `:invoke`/`:invoke_modify` statements whose target is still a bare `MethodInstance`
  (inlining's `compileable_specialization` emits those when the compileable
  specialization was not cached at optimization time; codegen lowers them to runtime
  dispatch) are resolved to a `CodeInstance` — running inference through `interp` if
  the cache has none — and the statement is rewritten to target it, in a copy of the
  containing source; cached `CodeInfo` is never mutated. Only targets callable through
  a native ABI (concrete signature, fully-resolved sparams) are resolved; anything
  else (e.g. `@nospecialize`-widened compileable signatures) deliberately keeps its
  runtime-dispatch fallback semantics.
- Referenced `CodeInstance`s that lack stored source (e.g. cached by an earlier
  session) are re-inferred.

`interp` must match the cache owner and world that produced `ci` (typically the
interpreter previously passed to [`typeinf!`](@ref)).

On Julia 1.11, code generation resolves callees through a lookup callback instead, so
only the root entry is returned.
"""
get_codeinfos(interp::CC.AbstractInterpreter, ci::Core.CodeInstance) =
    collect_codeinfos(interp, ci, nothing)

function collect_codeinfos(interp::CC.AbstractInterpreter,
                           root::Core.CodeInstance, root_src::Union{Core.CodeInfo, Nothing})
    codeinfos = Pair{Core.CodeInstance, Core.CodeInfo}[]
    @static if VERSION >= v"1.12-"
        visited = IdSet{Core.CodeInstance}()
        workqueue = Core.CodeInstance[root]
        while !isempty(workqueue)
            callee_ci = pop!(workqueue)
            callee_ci in visited && continue
            push!(visited, callee_ci)

            src = callee_ci === root && root_src !== nothing ? root_src :
                                                               get_source(callee_ci)
            if src === nothing
                # a referenced CI may lack stored source (e.g. it was cached by an
                # earlier session whose sources were dropped); re-establish it
                typeinf!(interp, CC.get_ci_mi(callee_ci))
                src = get_source(callee_ci)
                # if inference cannot provide source either, leave the call site to
                # codegen's runtime-dispatch fallback
                src === nothing && continue
            end
            src = resolve_invoke_targets(interp, src)
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
        src = root_src === nothing ? get_source(root) : root_src
        src !== nothing && push!(codeinfos, root => src)
    end
    return codeinfos
end

@static if VERSION >= v"1.12-"
# Mirror of nightly's `Compiler.has_valid_abi_sparams`: specializations with
# incomplete sparams (TypeVar) or SimpleVector/Vararg sparams cannot be called
# through a native specsig ABI — codegen's `needsparams` path emits `jl_invoke`
# even for CodeInstance operands, so rewriting such targets is useless.
function has_valid_abi_sparams(mi::Core.MethodInstance)
    for sp in mi.sparam_vals
        if sp isa TypeVar || sp isa Core.SimpleVector || CC.isvarargtype(sp)
            return false
        end
    end
    return true
end

# Rewrite `:invoke`/`:invoke_modify` statements whose target is still a bare
# `MethodInstance` to target a `CodeInstance` instead, inferring one when the cache
# has none. Codegen can only emit a direct call for a `CodeInstance` operand; a
# `MethodInstance` operand unconditionally lowers to runtime dispatch. Returns `src`
# unchanged when there is nothing to rewrite, or a rewritten copy (cached source is
# never mutated).
function resolve_invoke_targets(interp::CC.AbstractInterpreter, src::Core.CodeInfo)
    resolved = src
    for pc in eachindex(resolved.code)
        stmt = resolved.code[pc]
        rhs = stmt isa Expr && stmt.head === :(=) ? stmt.args[2] : stmt
        rhs isa Expr && (rhs.head === :invoke || rhs.head === :invoke_modify) || continue

        callee_mi = rhs.args[1]
        callee_mi isa Core.MethodInstance || continue
        # Only specializations with a fully-concrete signature can be invoked directly
        # through a native ABI; anything else (e.g. `@nospecialize`-widened compileable
        # signatures) keeps its runtime-dispatch fallback semantics.
        callee_mi.def isa Method && isdispatchtuple(callee_mi.specTypes) &&
            has_valid_abi_sparams(callee_mi) || continue

        callee_ci = typeinf!(interp, callee_mi)
        callee_ci === nothing && continue
        # only rewrite when the callee's source is available, so the returned
        # collection stays closed under direct-call edges
        get_source(callee_ci) === nothing && continue

        if resolved === src
            resolved = copy(src)
            stmt = resolved.code[pc]
            rhs = stmt isa Expr && stmt.head === :(=) ? stmt.args[2] : stmt
        end
        new_rhs = Expr(rhs.head, callee_ci, rhs.args[2:end]...)
        resolved.code[pc] = stmt === rhs ? new_rhs : Expr(:(=), stmt.args[1], new_rhs)
    end
    return resolved
end
end

"""
    get_codeinfos(interp::AbstractInterpreter, ci::CodeInstance, argtypes::Vector{Any}) ->
        Vector{Pair{CodeInstance, CodeInfo}}

Const-specialized variant of [`get_codeinfos(interp, ci)`](@ref): the callee walk is
seeded with the const-optimized source stored for `argtypes` (see
[`typeinf!(cache, interp, mi, argtypes)`](@ref)), so callees reachable only from the
const-optimized code are included as well.

Falls back to the generic source if no const entry exists for the given argtypes.
"""
get_codeinfos(interp::CC.AbstractInterpreter, ci::Core.CodeInstance, argtypes::Vector{Any}) =
    collect_codeinfos(interp, ci, get_source(ci, argtypes))

end # @static if VERSION >= v"1.11"

end # module CompilerCaching
