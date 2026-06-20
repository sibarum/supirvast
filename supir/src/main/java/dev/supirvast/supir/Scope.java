package dev.supirvast.supir;

import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.PushConstants;
import dev.supirvast.vastir.core.Texture;

import java.util.HashMap;
import java.util.Map;

/**
 * The Supir symbol table — resolves the names a form refers to back to the by-identity {@code core} entities
 * they were declared as. Chained: a module scope holds functions and module-level resources, a function scope
 * adds params and that function's resources, and each {@code if}/{@code while} region nests a child scope for
 * its locals. Lookups walk up to the root; definitions land in the current scope.
 *
 * <p>Resources (interface vars, buffers, textures, push-constant members) live in whatever scope they were
 * declared, so a resource declared in a function prelude is visible throughout that function's regions.
 */
final class Scope {

    private final Scope parent;

    private final Map<String, LocalVar> locals = new HashMap<>();
    private final Map<String, Expr.Param> params = new HashMap<>();
    private final Map<String, InterfaceVar> interfaces = new HashMap<>();
    private final Map<String, Buffer> buffers = new HashMap<>();
    private final Map<String, Texture> textures = new HashMap<>();
    private final Map<String, Function> functions = new HashMap<>();

    // At most one push-constant block is visible (the SPIR-V rule); members are read by name.
    private PushConstants pushConstants;
    private final Map<String, Integer> pushConstantMembers = new HashMap<>();

    Scope() {
        this.parent = null;
    }

    Scope(Scope parent) {
        this.parent = parent;
    }

    Scope child() {
        return new Scope(this);
    }

    // --- definitions (land in this scope) -------------------------------------------------------------------

    void defineLocal(String name, LocalVar var, Span at) {
        if (locals.containsKey(name)) {
            throw new SupirParseException(at, "duplicate local '" + name + "'");
        }
        locals.put(name, var);
    }

    void defineParam(String name, Expr.Param param, Span at) {
        if (params.containsKey(name)) {
            throw new SupirParseException(at, "duplicate parameter '" + name + "'");
        }
        params.put(name, param);
    }

    void defineInterface(String name, InterfaceVar var, Span at) {
        requireFreeResource(name, at);
        interfaces.put(name, var);
    }

    void defineBuffer(String name, Buffer buffer, Span at) {
        requireFreeResource(name, at);
        buffers.put(name, buffer);
    }

    void defineTexture(String name, Texture texture, Span at) {
        requireFreeResource(name, at);
        textures.put(name, texture);
    }

    void defineFunction(String name, Function function, Span at) {
        if (functions.containsKey(name)) {
            throw new SupirParseException(at, "duplicate function '" + name + "'");
        }
        functions.put(name, function);
    }

    void definePushConstants(PushConstants block, Span at) {
        if (rootPushConstants() != null) {
            throw new SupirParseException(at, "a module may declare at most one push-constant block");
        }
        this.pushConstants = block;
        for (int i = 0; i < block.members().size(); i++) {
            pushConstantMembers.put(block.members().get(i).name(), i);
        }
    }

    private void requireFreeResource(String name, Span at) {
        if (interfaces.containsKey(name) || buffers.containsKey(name) || textures.containsKey(name)) {
            throw new SupirParseException(at, "duplicate resource '" + name + "'");
        }
    }

    // --- lookups (walk up to the root) ----------------------------------------------------------------------

    LocalVar local(String name) {
        for (Scope s = this; s != null; s = s.parent) {
            LocalVar v = s.locals.get(name);
            if (v != null) {
                return v;
            }
        }
        return null;
    }

    Expr.Param param(String name) {
        for (Scope s = this; s != null; s = s.parent) {
            Expr.Param p = s.params.get(name);
            if (p != null) {
                return p;
            }
        }
        return null;
    }

    InterfaceVar iface(String name) {
        for (Scope s = this; s != null; s = s.parent) {
            InterfaceVar v = s.interfaces.get(name);
            if (v != null) {
                return v;
            }
        }
        return null;
    }

    Buffer buffer(String name) {
        for (Scope s = this; s != null; s = s.parent) {
            Buffer b = s.buffers.get(name);
            if (b != null) {
                return b;
            }
        }
        return null;
    }

    Texture texture(String name) {
        for (Scope s = this; s != null; s = s.parent) {
            Texture t = s.textures.get(name);
            if (t != null) {
                return t;
            }
        }
        return null;
    }

    Function function(String name) {
        for (Scope s = this; s != null; s = s.parent) {
            Function f = s.functions.get(name);
            if (f != null) {
                return f;
            }
        }
        return null;
    }

    PushConstants pushConstants() {
        return rootPushConstants();
    }

    /** The index of push-constant member {@code name}, or {@code -1} if no block declares it. */
    int pushConstantMember(String name) {
        for (Scope s = this; s != null; s = s.parent) {
            Integer i = s.pushConstantMembers.get(name);
            if (i != null) {
                return i;
            }
        }
        return -1;
    }

    private PushConstants rootPushConstants() {
        for (Scope s = this; s != null; s = s.parent) {
            if (s.pushConstants != null) {
                return s.pushConstants;
            }
        }
        return null;
    }
}
