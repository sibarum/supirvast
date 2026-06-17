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

    /** Reads {@code buffer[index]} (an i32 element of a storage buffer). */
    record BufferLoad(Buffer buffer, Expr index) implements Expr {
        @Override
        public Type type() {
            return Type.int32();
        }
    }

    /** A binary operation; comparison/logical ops yield {@code bool}, arithmetic yields the operand type. */
    record Binary(BinaryOp op, Expr lhs, Expr rhs) implements Expr {
        @Override
        public Type type() {
            return op.producesBool() ? Type.BOOL : lhs.type();
        }
    }

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
}
