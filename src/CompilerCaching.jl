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
# Global compile hook for debugging/inspection/reflection
#==============================================================================#

export compile_hook, compile_hook!

const _COMPILE_HOOK = Ref{Union{Nothing, Function}}(nothing)

"""
    compile_hook() -> Union{Nothing, Function}

Get the current compile hook.
"""
compile_hook() = _COMPILE_HOOK[]

"""
    compile_hook!(f)
    compile_hook!(nothing)

Set the global compile hook. Called at the start of every `cached_compilation`
invocation with `(cache, mi, world)`. Return value ignored.

The hook is called even for fully cached calls that don't re-link.
"""
compile_hook!(f) = _COMPILE_HOOK[] = f

#==============================================================================#
# CacheOwner - identifies a cache partition
#==============================================================================#

"""
    CacheOwner{K}

Identifies a cache partition. Constructed internally from a tag and optional keys.

- `tag::Symbol` - Base identifier (e.g., `:SynchJulia`, `:GPUCompiler`)
- `keys::K` - Additional sharding keys (e.g., device capability, optimization level)
"""
struct CacheOwner{K}
    tag::Symbol
    keys::K
end

function Base.hash(o::CacheOwner, h::UInt)
    h = hash(o.tag, h)
    h = hash(o.keys, h)
    return h
end

Base.:(==)(a::CacheOwner, b::CacheOwner) = a.tag == b.tag && a.keys == b.keys

#==============================================================================#
# CachedCompilationResult - wrapper for analysis_results storage
#==============================================================================#

"""
    CachedCompilationResult

Mutable wrapper type to identify our compilation results in the `analysis_results` chain.
This allows multiple compiler plugins to store results on the same CodeInstance.

Fields:
- `inferred::Any` - Result from inference phase (e.g., codeinfos vector)
- `linked::Any` - Result from link phase (e.g., function pointer)

The wrapper is created empty during `cache!` and populated via direct field access.
"""
mutable struct CachedCompilationResult
    inferred::Any
    linked::Any
end
CachedCompilationResult() = CachedCompilationResult(nothing, nothing)

"""
    initialize_result!(caller::CC.InferenceResult)

Create an empty `CachedCompilationResult` wrapper during inference.
The wrapper will be transferred to the CodeInstance and can later be
populated via direct field access after the link phase.
"""
function initialize_result!(caller::CC.InferenceResult)
    CC.stack_analysis_result!(caller, CachedCompilationResult())
end

"""
    get_result(ci::CodeInstance) -> Union{CachedCompilationResult, Nothing}

Retrieve the `CachedCompilationResult` wrapper from a CodeInstance's `analysis_results` chain.
Returns `nothing` if no wrapper is found.
"""
function get_result(ci::Core.CodeInstance)
    CC.traverse_analysis_results(ci) do @nospecialize result
        result isa CachedCompilationResult ? result : nothing
    end
end

"""
    set_result!(ci::CodeInstance, result)

Populate the `CachedCompilationResult` wrapper's `.linked` field with the compilation result.
The wrapper must have been created during `cache!`.
"""
function set_result!(ci::Core.CodeInstance, result)
    wrapper = get_result(ci)
    @assert wrapper !== nothing "CodeInstance without CachedCompilationResult wrapper; Please use `@setup_caching`."
    wrapper.linked = result
    return
end

#==============================================================================#
# CompilerCache - main entry point
#==============================================================================#

export CompilerCache, @setup_caching, cached_inference, cached_compilation

"""
    CompilerCache{K}

A compilation cache instance, parameterized by key type K.

- `tag::Symbol` - Base tag for cache owner
- `keys::K` - Sharding keys (e.g., device capability, optimization level)
"""
struct CompilerCache{K}
    tag::Symbol
    keys::K
end

# Non-parameterized constructor for K=Nothing
CompilerCache(tag::Symbol) = CompilerCache{Nothing}(tag, nothing)

"""
    cache_owner(cache::CompilerCache) -> CacheOwner

Get the CacheOwner for a cache. Use this as your interpreter's cache token
with `CC.cache_owner(interp)`.
"""
cache_owner(cache::CompilerCache{K}) where K = CacheOwner{K}(cache.tag, cache.keys)

"""
    @setup_caching InterpreterType.cache_field

Generate the required methods for an AbstractInterpreter to work with CompilerCaching.
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

    # Generate the appropriate ipo_dataflow_analysis! signature based on Julia version
    ipo_method = if hasmethod(CC.ipo_dataflow_analysis!, Tuple{CC.AbstractInterpreter, CC.OptimizationState, CC.IRCode, CC.InferenceResult})
        # Julia 1.12+: (interp, opt, ir, result)
        quote
            function $CC.ipo_dataflow_analysis!(interp::$InterpType, opt::$CC.OptimizationState,
                                                ir::$CC.IRCode, caller::$CC.InferenceResult)
                $initialize_result!(caller)
                @invoke $CC.ipo_dataflow_analysis!(interp::$CC.AbstractInterpreter, opt::$CC.OptimizationState,
                                                   ir::$CC.IRCode, caller::$CC.InferenceResult)
            end
        end
    else
        # Julia 1.11: (interp, ir, result)
        quote
            function $CC.ipo_dataflow_analysis!(interp::$InterpType, ir::$CC.IRCode,
                                                caller::$CC.InferenceResult)
                $initialize_result!(caller)
                @invoke $CC.ipo_dataflow_analysis!(interp::$CC.AbstractInterpreter, ir::$CC.IRCode,
                                                   caller::$CC.InferenceResult)
            end
        end
    end

    quote
        $CC.cache_owner(interp::$InterpType) = $cache_owner(interp.$cache_field)
        $ipo_method
    end |> esc
end


#==============================================================================#
# Method registration
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
# Inference helpers
#==============================================================================#

export populate!, cache!

"""
    populate!(cache, interp, mi) -> Vector{Pair{CodeInstance, Union{CodeInfo, Nothing}}}

Populate the code cache with CodeInstances for `mi` and its callees.

Runs type inference on `mi` using the provided interpreter, which must implement
the `CC.AbstractInterpreter` interface. The resulting CodeInstances are stored
in the cache for later retrieval.

Returns a vector of (CodeInstance, IR) pairs where:
- Native 1.12+: `[ci => CodeInfo, ...]` for root + callees
- Native 1.11: `[ci => nothing]` (codegen uses callback-based path)

The root CI is always the first entry: `first(result)[1]`
"""
function populate!(cache::CompilerCache, interp::CC.AbstractInterpreter,
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
        world = @static if isdefined(CC, :get_inference_world)
            CC.get_inference_world(interp)
        else
            CC.get_world_counter(interp)
        end
        ci = lookup(cache, mi; world)
        @assert ci !== nothing "Inference of $mi failed"
        if ci.inferred === nothing
            @atomic ci.inferred = src
        end

        # Return consistent type: [ci => nothing] (codegen uses callback path)
        return Pair{Core.CodeInstance, Union{Core.CodeInfo, Nothing}}[ci => nothing]
    end
end

#==============================================================================#
# Internal cache helpers
#==============================================================================#

@static if VERSION >= v"1.14-"
    function code_cache(owner::CacheOwner, world::UInt)
        world_range = CC.WorldRange(world)
        return CC.InternalCodeCache(owner, world_range)
    end
else
    function code_cache(owner::CacheOwner, world::UInt)
        cache = CC.InternalCodeCache(owner)
        return CC.WorldView(cache, world)
    end
end

function lookup(cache::CompilerCache, mi::Core.MethodInstance;
                world::UInt=Base.get_world_counter())
    owner = cache_owner(cache)
    cc = code_cache(owner, world)
    return CC.get(cc, mi, nothing)
end

"""
    store_backedges(mi::MethodInstance, ci::CodeInstance, deps::Vector{MethodInstance})

Register backedges so Julia automatically invalidates cached code when dependencies change.
This enables Julia's built-in invalidation mechanism - when any dependency MI is
invalidated, the caller MI's CodeInstances will have their max_world reduced.

Note: The API changed between Julia versions:
- Julia 1.11: jl_method_instance_add_backedge takes MethodInstance as caller
- Julia 1.12+: jl_method_instance_add_backedge takes CodeInstance as caller
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
    cache!(cache, mi; world, deps) -> CodeInstance

Create and store a CodeInstance for `mi` in the cache.

Used for foreign mode where inference doesn't run. The CI participates in
Julia's invalidation mechanism via backedges registered from `deps`.

# Arguments
- `cache::CompilerCache` - The compiler cache instance
- `mi::MethodInstance` - The method instance to cache
- `world::UInt` - World age for the CI
- `deps::Vector{MethodInstance}` - Dependencies to register as backedges
"""
function cache!(cache::CompilerCache, mi::Core.MethodInstance;
                world::UInt=Base.get_world_counter(),
                deps::Vector{Core.MethodInstance}=Core.MethodInstance[])
    owner = cache_owner(cache)
    cc = code_cache(owner, world)
    edges = isempty(deps) ? Core.svec() : Core.svec(deps...)

    # Create empty wrapper for later population via `set_result!`
    ar = CC.AnalysisResults(CachedCompilationResult(), CC.NULL_ANALYSIS_RESULTS)

    @static if VERSION >= v"1.12-"
        ci = Core.CodeInstance(mi, owner, Any, Any, nothing, nothing,
            Int32(0), UInt(world), typemax(UInt), UInt32(0), ar, nothing, edges)
    else
        ci = Core.CodeInstance(mi, owner, Any, Any, nothing, nothing,
            Int32(0), UInt(world), typemax(UInt), UInt32(0), UInt32(0), ar, UInt8(0))
    end
    CC.setindex!(cc, ci, mi)

    # Register backedges for automatic invalidation
    if !isempty(deps)
        store_backedges(mi, ci, deps)
    end

    return ci
end

#==============================================================================#
# Cached inference - dependency discovery without full compilation
#==============================================================================#

"""
    cached_inference(cache, mi, world; infer) -> (CodeInstance, result_or_nothing)

Run only the inference phase for a method instance without codegen or link.

This is useful when recursively processing dependencies during the infer phase:
you want to establish the dependency tracking (creating CodeInstances with backedges)
without running codegen/link for each callee.

The `infer` callback must return a tuple `(ci::CodeInstance, result::Any)` where
`result` is opaque to the caching layer and stored for later retrieval.

Returns `(ci, result)` where `result` is from the infer callback,
or `nothing` if the CI was already cached without a stored result.
"""
function cached_inference(cache::CompilerCache, mi::Core.MethodInstance,
                          world::UInt; infer)
    ci = lookup(cache, mi; world)
    if ci !== nothing
        # Check if we have cached infer result
        wrapper = get_result(ci)
        if wrapper !== nothing && wrapper.inferred !== nothing
            return ci, wrapper.inferred
        end
        # CI exists but no cached infer result (e.g., Julia-created CI)
        return ci, nothing
    end

    ci, result = infer(cache, mi, world)  # Simple tuple destructure
    @assert ci !== nothing "Inference failed to produce a CodeInstance"

    # Store infer result for cache hit retrieval
    wrapper = get_result(ci)
    if wrapper !== nothing
        wrapper.inferred = result
    end

    return ci, result
end

#==============================================================================#
# Cached compilation - main API
#==============================================================================#

"""
    cached_compilation(cache, mi, world; infer, codegen, link) -> result

Three-phase cached compilation with automatic invalidation.
"""
function cached_compilation(cache::CompilerCache, mi::Core.MethodInstance,
                            world::UInt; infer, codegen, link)
    # Call compile hook if set (even on cache hit)
    hook = compile_hook()
    if hook !== nothing
        hook(cache, mi, world)
    end

    # 1. Run inference phase (checks cache internally, returns early if CI exists)
    ci, inferred = cached_inference(cache, mi, world; infer)

    # 2. Check for cached linked result
    wrapper = get_result(ci)
    if wrapper !== nothing && wrapper.linked !== nothing
        return wrapper.linked
    end

    # 3. Run codegen
    # Need inferred result for codegen - re-run infer if we had a cache hit on CI
    if inferred === nothing
        _, inferred = infer(cache, mi, world)
    end

    compiled = codegen(cache, mi, world, inferred)

    # 4. Link and store result
    linked = link(cache, mi, world, compiled)
    set_result!(ci, linked)

    return linked
end

end # module CompilerCaching
