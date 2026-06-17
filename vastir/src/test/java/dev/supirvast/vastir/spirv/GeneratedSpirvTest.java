package dev.supirvast.vastir.spirv;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Sanity checks that the grammar-generated SPIR-V types behave correctly at runtime, not just compile. */
class GeneratedSpirvTest {

    @Test
    void opcodeLookupRoundTrips() {
        assertEquals(57, Op.OpFunctionCall.opcode());
        assertSame(Op.OpFunctionCall, Op.byOpcode(57).orElseThrow());
    }

    @Test
    void operandLayoutIsTypedAndOrdered() {
        List<Operand> operands = Op.OpFunctionCall.operands();
        assertEquals(4, operands.size());
        assertEquals(OperandKind.IdResultType, operands.get(0).kind());
        // Trailing call arguments are variadic.
        Operand last = operands.get(operands.size() - 1);
        assertEquals(OperandKind.IdRef, last.kind());
        assertEquals(Quantifier.VARIADIC, last.quantifier());
    }

    @Test
    void operandlessOpHasEmptyLayout() {
        assertTrue(Op.OpNop.operands().isEmpty());
    }

    @Test
    void valueEnumRoundTrips() {
        assertSame(StorageClass.Input, StorageClass.byValue(StorageClass.Input.value()).orElseThrow());
    }

    @Test
    void sanitizedNamesKeepGrammarValues() {
        // Dim."1D" became _1D but must keep value 0.
        assertEquals(0, Dim._1D.value());
    }

    @Test
    void bitEnumUsesMaskValues() {
        assertEquals(0x0002, FunctionControl.DontInline.value());
    }

    @Test
    void parameterizedEnumerantCarriesOperands() {
        assertEquals(3, ExecutionMode.LocalSize.parameters().size());
        assertEquals(OperandKind.LiteralInteger, ExecutionMode.LocalSize.parameters().get(0).kind());
    }

    @Test
    void operandKindKnowsItsCategory() {
        assertEquals(OperandKind.Category.ID, OperandKind.IdRef.category());
        assertEquals(OperandKind.Category.VALUE_ENUM, OperandKind.StorageClass.category());
        assertEquals(OperandKind.Category.BIT_ENUM, OperandKind.FunctionControl.category());
    }
}
