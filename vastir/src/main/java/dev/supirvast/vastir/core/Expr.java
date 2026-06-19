package dev.supirvast.vastir.core;

import dev.supirvast.vastir.type.Type;

import java.util.List;

/**
 * A core-level, value-producing expression. Expressions form trees (operands are expressions); the lowering
 * evaluates each to an SSA result id. Every expression knows its result {@link #type()}.
 */
public sealed interface Expr {

    Type type();

    /** A 32-bit integer constant of the given int type. The record's {@code type()} accessor (returning
     * {@code Type.Int}) covariantly satisfies {@link Expr#type()}. */
    record ConstInt(Type.Int type, long value) implements Expr {}

    /** A 32-bit floating-point constant; {@code type()} returns {@code Type.Float}, covariant with the interface. */
    record ConstFloat(Type.Float type, double value) implements Expr {}

    record ConstBool(boolean value) implements Expr {
        @Override
        public Type type() {
            return Type.BOOL;
        }
    }

    /** Reads the current value of a local variable. */
    record Read(LocalVar variable) implements Expr {
        @Override
        public Type type() {
            return variable.type();
        }
    }

    /** The x component of {@code gl_GlobalInvocationID} — this invocation's index in the dispatch. */
    record InvocationId() implements Expr {
        @Override
        public Type type() {
            return Type.int32();
        }
    }

    /** Reads {@code buffer[index]} — an element of a storage buffer, typed by the buffer's element type. */
    record BufferLoad(Buffer buffer, Expr index) implements Expr {
        @Override
        public Type type() {
            return buffer.element();
        }
    }

    /** Reads a graphics-pipeline built-in input (e.g. {@code gl_VertexIndex}); its type is fixed by the builtin. */
    record BuiltinRead(Builtin builtin) implements Expr {
        @Override
        public Type type() {
            return builtin.type();
        }
    }

    /** Reads a stage interface input variable at its declared {@code location}. */
    record InterfaceRead(InterfaceVar variable) implements Expr {
        @Override
        public Type type() {
            return variable.type();
        }
    }

    /** A binary operation; comparison/logical ops yield {@code bool}, arithmetic yields the operand type. */
    record Binary(BinaryOp op, Expr lhs, Expr rhs) implements Expr {
        @Override
        public Type type() {
            return op.producesBool() ? Type.BOOL : lhs.type();
        }
    }

    /**
     * Reinterprets {@code operand}'s bits as {@code type}, without changing them — the core-level
     * {@code OpBitcast}. Used to move between {@code i32} and {@code uint32} (same machine representation, so
     * a no-op on the CPU) so that signed and unsigned operators can be selected explicitly: load an i32 buffer
     * element, bitcast it to {@code uint32} to drive unsigned arithmetic, then bitcast the result back to
     * {@code i32} to store it. Operand and target must have the same bit width.
     */
    record Bitcast(Expr operand, Type type) implements Expr {}

    /**
     * Converts {@code operand} to a different-width integer {@code type} — the core-level {@code OpSConvert}/
     * {@code OpUConvert}. Extension when widening follows the <em>target</em> type's signedness (signed target
     * sign-extends, unsigned target zero-extends), since SPIR-V ties the opcode to the result type; narrowing
     * truncates to the low bits. This is what moves between {@code i32} and {@code i64} (e.g. widen an i32
     * buffer element before 64-bit arithmetic, then narrow the result back to store it). Operand and target
     * must have <em>different</em> bit widths (SPIR-V forbids a same-width convert; use {@link Bitcast} to
     * change signedness at the same width).
     */
    record Convert(Expr operand, Type type) implements Expr {}

    /** A unary operation. */
    record Unary(UnaryOp op, Expr operand) implements Expr {
        @Override
        public Type type() {
            return op == UnaryOp.LOGICAL_NOT ? Type.BOOL : operand.type();
        }
    }

    /** Builds a vector from its components. The record's {@code type()} accessor (a {@code Type.Vector})
     * covariantly satisfies {@link Expr#type()}. */
    record VectorConstruct(Type.Vector type, List<Expr> components) implements Expr {
        public VectorConstruct {
            components = List.copyOf(components);
        }
    }

    /** Extracts component {@code index} of a vector. */
    record VectorExtract(Expr vector, int index) implements Expr {
        @Override
        public Type type() {
            return ((Type.Vector) vector.type()).component();
        }
    }

    /** Reads the {@code index}-th parameter of the enclosing function (carries its own type for context). */
    record Param(int index, Type type) implements Expr {
    }

    /** Calls another function with the given arguments; its type is the callee's return type. */
    record Call(Function callee, List<Expr> arguments) implements Expr {
        public Call {
            arguments = List.copyOf(arguments);
        }

        @Override
        public Type type() {
            return callee.signature().returnType();
        }
    }

    /**
     * A built-in floating-point math intrinsic ({@link MathFn}) over scalar/vector operands — {@code dot},
     * {@code normalize}, {@code pow}, {@code clamp}, etc. The result {@code type} is carried explicitly; use the
     * factory helpers, which derive it ({@code DOT}/{@code LENGTH} reduce to the vector's component type, the
     * rest share the first argument's type). Lowers to {@code OpDot} or a {@code GLSL.std.450} {@code OpExtInst}.
     */
    record MathCall(MathFn fn, Type type, List<Expr> args) implements Expr {
        public MathCall {
            args = List.copyOf(args);
        }

        public static MathCall dot(Expr a, Expr b) {
            return new MathCall(MathFn.DOT, componentOf(a), List.of(a, b));
        }

        public static MathCall length(Expr v) {
            return new MathCall(MathFn.LENGTH, componentOf(v), List.of(v));
        }

        public static MathCall normalize(Expr v) {
            return new MathCall(MathFn.NORMALIZE, v.type(), List.of(v));
        }

        public static MathCall abs(Expr v) {
            return new MathCall(MathFn.ABS, v.type(), List.of(v));
        }

        public static MathCall sqrt(Expr v) {
            return new MathCall(MathFn.SQRT, v.type(), List.of(v));
        }

        public static MathCall inverseSqrt(Expr v) {
            return new MathCall(MathFn.INVERSE_SQRT, v.type(), List.of(v));
        }

        public static MathCall cross(Expr a, Expr b) {
            return new MathCall(MathFn.CROSS, a.type(), List.of(a, b));
        }

        public static MathCall reflect(Expr incident, Expr normal) {
            return new MathCall(MathFn.REFLECT, incident.type(), List.of(incident, normal));
        }

        public static MathCall pow(Expr base, Expr exponent) {
            return new MathCall(MathFn.POW, base.type(), List.of(base, exponent));
        }

        public static MathCall min(Expr a, Expr b) {
            return new MathCall(MathFn.MIN, a.type(), List.of(a, b));
        }

        public static MathCall max(Expr a, Expr b) {
            return new MathCall(MathFn.MAX, a.type(), List.of(a, b));
        }

        public static MathCall clamp(Expr x, Expr lo, Expr hi) {
            return new MathCall(MathFn.CLAMP, x.type(), List.of(x, lo, hi));
        }

        public static MathCall mix(Expr a, Expr b, Expr t) {
            return new MathCall(MathFn.MIX, a.type(), List.of(a, b, t));
        }

        // --- the rest of the float library (all elementwise, result shares the first argument's type) ------

        public static MathCall sin(Expr v) {
            return unary(MathFn.SIN, v);
        }

        public static MathCall cos(Expr v) {
            return unary(MathFn.COS, v);
        }

        public static MathCall tan(Expr v) {
            return unary(MathFn.TAN, v);
        }

        public static MathCall asin(Expr v) {
            return unary(MathFn.ASIN, v);
        }

        public static MathCall acos(Expr v) {
            return unary(MathFn.ACOS, v);
        }

        public static MathCall atan(Expr v) {
            return unary(MathFn.ATAN, v);
        }

        public static MathCall atan2(Expr y, Expr x) {
            return new MathCall(MathFn.ATAN2, y.type(), List.of(y, x));
        }

        public static MathCall sinh(Expr v) {
            return unary(MathFn.SINH, v);
        }

        public static MathCall cosh(Expr v) {
            return unary(MathFn.COSH, v);
        }

        public static MathCall tanh(Expr v) {
            return unary(MathFn.TANH, v);
        }

        public static MathCall asinh(Expr v) {
            return unary(MathFn.ASINH, v);
        }

        public static MathCall acosh(Expr v) {
            return unary(MathFn.ACOSH, v);
        }

        public static MathCall atanh(Expr v) {
            return unary(MathFn.ATANH, v);
        }

        public static MathCall exp(Expr v) {
            return unary(MathFn.EXP, v);
        }

        public static MathCall log(Expr v) {
            return unary(MathFn.LOG, v);
        }

        public static MathCall exp2(Expr v) {
            return unary(MathFn.EXP2, v);
        }

        public static MathCall log2(Expr v) {
            return unary(MathFn.LOG2, v);
        }

        public static MathCall round(Expr v) {
            return unary(MathFn.ROUND, v);
        }

        public static MathCall roundEven(Expr v) {
            return unary(MathFn.ROUND_EVEN, v);
        }

        public static MathCall trunc(Expr v) {
            return unary(MathFn.TRUNC, v);
        }

        public static MathCall floor(Expr v) {
            return unary(MathFn.FLOOR, v);
        }

        public static MathCall ceil(Expr v) {
            return unary(MathFn.CEIL, v);
        }

        public static MathCall fract(Expr v) {
            return unary(MathFn.FRACT, v);
        }

        public static MathCall sign(Expr v) {
            return unary(MathFn.SIGN, v);
        }

        public static MathCall radians(Expr degrees) {
            return unary(MathFn.RADIANS, degrees);
        }

        public static MathCall degrees(Expr radians) {
            return unary(MathFn.DEGREES, radians);
        }

        /** {@code step(edge, x)} — 0 below {@code edge}, 1 at/above it. Operands share a type. */
        public static MathCall step(Expr edge, Expr x) {
            return new MathCall(MathFn.STEP, edge.type(), List.of(edge, x));
        }

        /** {@code smoothstep(edge0, edge1, x)} — a smooth 0→1 Hermite ramp. Operands share a type. */
        public static MathCall smoothstep(Expr edge0, Expr edge1, Expr x) {
            return new MathCall(MathFn.SMOOTHSTEP, edge0.type(), List.of(edge0, edge1, x));
        }

        /** {@code fma(a, b, c) = a*b + c} (fused). Operands share a type. */
        public static MathCall fma(Expr a, Expr b, Expr c) {
            return new MathCall(MathFn.FMA, a.type(), List.of(a, b, c));
        }

        /** {@code distance(a, b)} — the length of {@code a - b}; reduces to the scalar component type. */
        public static MathCall distance(Expr a, Expr b) {
            return new MathCall(MathFn.DISTANCE, componentOf(a), List.of(a, b));
        }

        /** {@code faceforward(n, i, nref)} — flips {@code n} to face away from {@code i}. */
        public static MathCall faceForward(Expr n, Expr i, Expr nref) {
            return new MathCall(MathFn.FACE_FORWARD, n.type(), List.of(n, i, nref));
        }

        /** {@code refract(i, n, eta)} — refraction of incident {@code i} through normal {@code n} ({@code eta}
         * scalar). Result shares {@code i}'s vector type. */
        public static MathCall refract(Expr i, Expr n, Expr eta) {
            return new MathCall(MathFn.REFRACT, i.type(), List.of(i, n, eta));
        }

        private static MathCall unary(MathFn fn, Expr v) {
            return new MathCall(fn, v.type(), List.of(v));
        }

        private static Type componentOf(Expr vector) {
            return ((Type.Vector) vector.type()).component();
        }
    }
}
