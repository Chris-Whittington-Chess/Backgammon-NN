//! Dice and a small seedable RNG (SPEC §4).
//!
//! For M0 we use a self-contained SplitMix64 generator so the crate builds
//! offline and self-play / tests are fully reproducible from a seed. It can be
//! swapped for the `rand` crate later without touching call sites.

/// Minimal seedable PRNG (SplitMix64). Deterministic given a seed.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create an RNG from a seed. The same seed always yields the same stream.
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    /// Next raw 64-bit value (SplitMix64).
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `1..=6` using rejection sampling (unbiased).
    #[inline]
    pub fn die(&mut self) -> u8 {
        loop {
            // Take the top 3 bits -> 0..=7, reject 6 and 7.
            let v = (self.next_u64() >> 61) as u8;
            if v < 6 {
                return v + 1;
            }
        }
    }

    /// Roll a pair of dice.
    pub fn roll(&mut self) -> Dice {
        Dice::new(self.die(), self.die())
    }
}

/// The result of rolling two dice.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Dice {
    a: u8,
    b: u8,
}

impl Dice {
    /// Construct from two die faces (`1..=6` each). Panics on out-of-range.
    pub fn new(a: u8, b: u8) -> Self {
        assert!((1..=6).contains(&a) && (1..=6).contains(&b), "die out of range");
        Dice { a, b }
    }

    /// The two die faces as rolled.
    pub fn pair(&self) -> (u8, u8) {
        (self.a, self.b)
    }

    /// True if both dice show the same value (a "double").
    pub fn is_double(&self) -> bool {
        self.a == self.b
    }

    /// The die values available to play this turn. A double yields four copies
    /// of the face; a non-double yields the two distinct faces.
    pub fn plays(&self) -> Vec<u8> {
        if self.is_double() {
            vec![self.a; 4]
        } else {
            vec![self.a, self.b]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dice_faces_always_in_range() {
        let mut rng = Rng::new(0xDEAD_BEEF);
        for _ in 0..10_000 {
            let d = rng.roll();
            let (a, b) = d.pair();
            assert!((1..=6).contains(&a) && (1..=6).contains(&b));
        }
    }

    #[test]
    fn same_seed_reproduces_stream() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(r1.roll(), r2.roll());
        }
    }

    #[test]
    fn doubles_play_four_times() {
        let d = Dice::new(5, 5);
        assert!(d.is_double());
        assert_eq!(d.plays(), vec![5, 5, 5, 5]);
    }

    #[test]
    fn non_doubles_play_twice() {
        let d = Dice::new(3, 1);
        assert!(!d.is_double());
        assert_eq!(d.plays(), vec![3, 1]);
    }

    #[test]
    fn die_distribution_is_roughly_uniform() {
        let mut rng = Rng::new(12345);
        let mut counts = [0u32; 7];
        let n = 60_000;
        for _ in 0..n {
            counts[rng.die() as usize] += 1;
        }
        // Each face expected ~10000; allow generous slack.
        for (face, &c) in counts.iter().enumerate().skip(1) {
            assert!(c > 9000 && c < 11000, "face {face} count {c} out of band");
        }
    }
}
