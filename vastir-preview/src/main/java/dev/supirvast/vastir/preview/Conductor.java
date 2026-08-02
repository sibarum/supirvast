package dev.supirvast.vastir.preview;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * A minimal cooperative main-thread scheduler — the Orchestration API's <b>Conductor</b> (first-cut spike). It
 * seats {@link Player}s, each with a cadence (a tick period in nanoseconds; {@code 0} = eager, i.e. every pass),
 * and drives them on the calling thread until all have retired.
 *
 * <p>A periodic Player ticks once its deadline passes; an <b>eager</b> Player — e.g. a vsync-paced render whose
 * tick blocks on present — paces the loop. When no eager Player remains, the loop sleeps until the next deadline
 * instead of spinning.
 *
 * <p><b>Spike scope.</b> The clock is {@code System.nanoTime} and cadences are fixed periods. The real Conductor
 * draws its tempo from hardware backpressure (the present/DAC pull) and virtual timers, its Players are Pontif
 * conduits, and its Instruments are effect sinks. This proves the mechanism: differently-cadenced concerns
 * cohabiting one main thread with no coupling between them.
 */
public final class Conductor {

    private static final class Seat {
        final Player player;
        final long periodNanos;
        long dueNanos;

        Seat(Player player, long periodNanos, long dueNanos) {
            this.player = player;
            this.periodNanos = periodNanos;
            this.dueNanos = dueNanos;
        }
    }

    private final List<Seat> seats = new ArrayList<>();

    /** Seat a Player with a cadence period in nanoseconds ({@code 0} = eager: ticks every pass). */
    public Conductor seat(Player player, long periodNanos) {
        seats.add(new Seat(player, periodNanos, System.nanoTime()));
        return this;
    }

    /** Drive the orchestra on the calling (main) thread until every Player has retired. */
    public void run() {
        while (!seats.isEmpty()) {
            long now = System.nanoTime();
            boolean anyEager = false;
            Iterator<Seat> it = seats.iterator();
            while (it.hasNext()) {
                Seat seat = it.next();
                if (seat.periodNanos == 0) {
                    anyEager = true;
                }
                if (now >= seat.dueNanos) {
                    if (!seat.player.play(now)) {
                        it.remove();
                    } else {
                        seat.dueNanos = seat.periodNanos == 0 ? now : now + seat.periodNanos;
                    }
                }
            }
            if (!anyEager && !seats.isEmpty()) {
                sleepUntilEarliest();   // nothing paces the loop; wait out the nearest deadline
            }
        }
    }

    private void sleepUntilEarliest() {
        long earliest = Long.MAX_VALUE;
        for (Seat seat : seats) {
            earliest = Math.min(earliest, seat.dueNanos);
        }
        long waitNanos = earliest - System.nanoTime();
        if (waitNanos > 0) {
            try {
                Thread.sleep(waitNanos / 1_000_000L, (int) (waitNanos % 1_000_000L));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
