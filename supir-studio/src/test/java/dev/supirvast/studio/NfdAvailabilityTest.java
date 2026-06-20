package dev.supirvast.studio;

import org.junit.jupiter.api.Test;
import sibarum.dasum.gui.natives.nfd.Nfd;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Guards the Export… folder picker: its native backend (nfd.dll) ships in {@code dasum-nfd}. If that runtime
 * dependency is dropped, {@link Nfd#ensureInit()} can't load the library and the dialog silently fails — this
 * catches that at build time. The native is Windows-only, matching the studio's target.
 */
class NfdAvailabilityTest {

    @Test
    void nativeFileDialogLibraryLoads() {
        assumeTrue(System.getProperty("os.name").toLowerCase().contains("win"),
                "the NFD native (nfd.dll) is bundled for Windows only");
        assertDoesNotThrow(Nfd::ensureInit,
                "nfd.dll failed to load — is the dasum-nfd runtime dependency present?");
    }
}
