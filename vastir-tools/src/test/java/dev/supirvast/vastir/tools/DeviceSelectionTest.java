package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.tools.DeviceSelection.Candidate;
import org.junit.jupiter.api.Test;

import java.util.List;

import static dev.supirvast.vastir.tools.DeviceSelection.CPU;
import static dev.supirvast.vastir.tools.DeviceSelection.DISCRETE;
import static dev.supirvast.vastir.tools.DeviceSelection.INTEGRATED;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Which device a context runs on, decided without one: the choice is a pure function of what is listed. */
class DeviceSelectionTest {

    /** This machine's order: the integrated GPU first, the discrete one second. */
    private static final List<Candidate> LAPTOP = List.of(
            new Candidate(0, "Intel(R) Graphics", INTEGRATED, true),
            new Candidate(1, "NVIDIA GeForce RTX 5070 Ti Laptop GPU", DISCRETE, true));

    @Test
    void theDiscreteGpuWinsWhereverItIsListed() {
        assertEquals(1, DeviceSelection.choose(LAPTOP, null));
        assertEquals(1, DeviceSelection.choose(LAPTOP, " "));
    }

    @Test
    void aDeviceWithoutComputeIsNeverChosen() {
        List<Candidate> devices = List.of(
                new Candidate(0, "Display only", DISCRETE, false),
                new Candidate(1, "Intel(R) Graphics", INTEGRATED, true));
        assertEquals(1, DeviceSelection.choose(devices, null));
        assertEquals(-1, DeviceSelection.choose(List.of(devices.getFirst()), null));
    }

    @Test
    void anIntegratedGpuBeatsASoftwareOne() {
        List<Candidate> devices = List.of(
                new Candidate(0, "llvmpipe", CPU, true),
                new Candidate(1, "Intel(R) Graphics", INTEGRATED, true));
        assertEquals(1, DeviceSelection.choose(devices, null));
    }

    @Test
    void theOverrideTakesATypeOrAPartOfTheName() {
        assertEquals(0, DeviceSelection.choose(LAPTOP, "integrated"));
        assertEquals(1, DeviceSelection.choose(LAPTOP, "Discrete"));
        assertEquals(0, DeviceSelection.choose(LAPTOP, "intel"));
        assertEquals(1, DeviceSelection.choose(LAPTOP, "RTX 5070"));
    }

    /** A benchmark asked for one GPU must not quietly run on another. */
    @Test
    void anOverrideThatMatchesNothingIsAnErrorNamingTheDevices() {
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> DeviceSelection.choose(LAPTOP, "radeon"));
        assertTrue(e.getMessage().contains("Intel(R) Graphics (integrated)"), e.getMessage());
        assertTrue(e.getMessage().contains("NVIDIA GeForce RTX 5070 Ti Laptop GPU (discrete)"), e.getMessage());
    }
}
