"""Group 3: population-aware selection, guarded commitment and endgame recovery.

Only public advertisements and this player's own date history are used.
Group 9's strategic ideas about willingness and remaining opportunities informed
our experiments; this implementation and its loss model were independently written.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from models.player import (
	Advertisement,
	Candidate,
	DateContext,
	DateOutcome,
	Decision,
	GameContext,
	PlayerSnapshot,
	RoundContext,
)
from models.player import Player as BasePlayer
from models.profile import MAX_VALUE, Profile, decrement


@dataclass
class _PartnerState:
	observations: list[Profile] = field(default_factory=list)
	stated_ideals: list[Profile] = field(default_factory=list)
	trust: float = 1.0
	proposals_to: int = 0
	dates_without_reciprocate: int = 0


class Player3(BasePlayer):
	"""The confirmed policy, with inactive experiments and wrapper layers removed."""

	MIN_DATES = 2

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self._commit_round = max(1, int(self.rounds * 0.65))
		self._est: dict[int, float] = {}
		self._est_lb: dict[int, float] = {}
		self._seen: list[float] = []
		self._proposed_to_me: set[int] = set()
		self._mutual_again: dict[int, int] = {}
		self._partners: dict[int, _PartnerState] = {}
		self._dim_sum = [0.0] * self.dims
		self._dim_count = 0
		self._last_projection = None
		self._ad = None
		self._last_singles = ctx.people
		self._sent_profiles = {}
		self._population = [[1.0] * 11 for _ in range(self.dims)]
		self._population_seen = set()
		self._loss_cache = None
		self._r2_variance_cache = None

	def advertise(self) -> Advertisement:
		showable = range(self.monotonic) if self.monotonic else range(self.dims)
		own = max(showable, key=lambda i: self.profile[i])
		value = min(MAX_VALUE, self.profile[own])
		if own < self.monotonic and value >= 0 and self.bonus / self.dims >= 0:
			value = max(value, min(10, value + 2))
		means = self._dim_means()
		seek = max(range(self.dims), key=lambda i: self._gap(i, self.ideal[i], means[i]))
		self._ad = Advertisement(own, value, seek, self.ideal[seek])
		return self._ad

	def project(self, date: DateContext) -> Profile:
		if date.partner not in self._sent_profiles:
			values = list(self.profile)
			if self._ad:
				values[self._ad.own_dim] = self._ad.own_value
			self._sent_profiles[date.partner] = tuple(values)
		self._last_projection = self._sent_profiles[date.partner]
		return self._last_projection

	def _lie_allowed(self, date, observed):
		state = self._partners.get(date.partner)
		if date.round < 0.35 * self.rounds or (state and state.trust < 0.75):
			return False
		ideal = state.stated_ideals[0] if state and state.stated_ideals else self._dim_means()
		return self._estimate(observed, ideal) > 0.1 * self.bonus

	def describe_ideal(self, date: DateContext, observed: Profile) -> Profile:
		values = list(self.ideal)
		blend = 0.65
		if self._lie_allowed(date, observed):
			for i, have in enumerate(observed):
				target = min(have, values[i]) if i < self.monotonic else have
				shift = round((1 - blend) * values[i] + blend * target) - values[i]
				values[i] += max(-10, min(10, shift))
		if self._ad:
			values[self._ad.seek_dim] = self._ad.seek_value
		# Keep the advertised preference fixed and avoid an exact deceptive mirror.
		if tuple(values) != self.ideal and tuple(values) == observed:
			editable = [i for i in range(self.dims) if not (self._ad and i == self._ad.seek_dim)]
			if editable:
				i = min(editable, key=lambda j: abs(self.ideal[j] - values[j]))
				values[i] += -1 if values[i] > 0 else 1
		return tuple(values)

	def _confidence_lb(self, est, trust):
		return est

	def _reservation_v1(self, rounds_left):
		frac = max(0.0, min(1.0, (rounds_left - 1) / max(1, self.rounds - 1)))
		base = self._percentile(self._seen, 0.70) if len(self._seen) >= 5 else 0.5 * self.bonus
		return max(0.0, base) * frac

	def _reservation(self, rounds_left, singles):
		frac = max(0.0, min(1.0, (rounds_left - 1) / max(1, self.rounds - 1)))
		base = self._percentile(self._seen, 0.70) if len(self._seen) >= 5 else 0.5 * self.bonus
		base *= 0.6 + 0.4 * (1.0 - singles / max(2, self.people))
		return max(0.0, base) * frac

	def _quality_bar(self, date):
		if date.rounds_left <= 3 or self._last_singles <= 4:
			return 0.0
		mine, theirs = self._prior_losses()
		prior = self.bonus - sum(mine) - sum(theirs) / 2
		values = list(self._est.values())
		ref = (4 * prior + sum(values)) / (4 + len(values))
		var = (4 * self._r2_prior_variance() + sum((v - ref) ** 2 for v in values)) / (
			4 + len(values)
		)
		horizon = min(1.0, (date.rounds_left - 2) / max(4, self.rounds / 4))
		return (
			max(0, ref + 1.5 * math.sqrt(var))
			* min(1, self._last_singles / max(2, self.people)) ** 0.125
			* horizon
		)

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		self._last_singles = turn.singles
		endgame = turn.rounds_left <= 3
		bar = self._quality_bar(turn)
		ranked = []
		for candidate in candidates:
			index = candidate.index
			if turn.rounds_left == 1 and not (candidate.dates and self._est.get(index, 0) > 0):
				continue
			if index in self._est:
				score = self._est[index]
				if score <= 0 or (not endgame and turn.singles > 4 and score <= bar):
					continue
				# Reopen previously exhausted offers only during the final three rounds.
				if (
					not endgame
					and self._partner(index).proposals_to >= 4
					and index not in self._proposed_to_me
				):
					continue
				if index in self._proposed_to_me:
					score += max(0.0, self.bonus - score) / 4
			else:
				score, _ = self._ad_utilities(candidate.ad)
			ranked.append((score, -index, index))
		ranked.sort(reverse=True)
		cap = min(32, self.list_limit) if self.list_limit is not None else 32
		return [entry[2] for entry in ranked[:cap]]

	def _date_decision(self, date, observed, stated_ideal):
		state = self._partner(date.partner)
		previous_proposals = state.proposals_to
		# Preserve the confirmed observation/update order, including its trust semantics.
		state.observations.append(observed)
		state.stated_ideals.append(stated_ideal)
		self._observe_dims(observed)
		adj_obs, adj_ideal, trust = self._adjust_profiles(
			date.partner,
			date.partner_ad,
			observed,
			stated_ideal,
			self._last_projection or self.profile,
		)
		est = self._estimate(adj_obs, adj_ideal, trust)
		lb = self._confidence_lb(est, trust)
		self._est[date.partner] = est
		self._est_lb[date.partner] = lb
		self._seen.append(est)
		decision = Decision()
		if not (lb <= 0 and est <= 0):
			bar = self._reservation(date.rounds_left, self._last_singles)
			offered = date.partner in self._proposed_to_me
			willing = (offered or self._mutual_again.get(date.partner, 0) >= 2) and lb >= 0.05
			propose = date.date_number >= 2 and (
				lb > bar or (willing and lb > 0) or (date.rounds_left <= 3 and lb > 0)
			)
			if offered and lb > 0:
				propose = propose or date.date_number >= 2
			exploring = date.rounds_left > max(3, int(self.rounds * 0.35))
			pursue = lb > max(bar, 0.5 * self.bonus) if exploring else lb > 0.0
			if propose:
				state.proposals_to += 1
			decision = Decision(again=pursue or propose, propose=propose)
		if not decision.again and date.rounds_left > 1 and state.trust >= 0.75:
			theirs = (
				self.bonus
				- decrement(stated_ideal, self.profile, self.monotonic)
				- decrement(self.ideal, observed, self.monotonic) / 2
			)
			if min(self._est[date.partner], theirs) > max(
				0.35 * self.bonus, self._reservation(date.rounds_left, self._last_singles)
			):
				decision = Decision(again=True, propose=decision.propose)
		if state.proposals_to >= 4 and date.partner not in self._proposed_to_me:
			decision = Decision(again=False, propose=decision.propose)
		if self._estimate(observed, stated_ideal) <= 0.0 * self.bonus:
			state.proposals_to = previous_proposals
			return Decision()
		return decision

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		if date.partner not in self._population_seen:
			self._population_seen.add(date.partner)
			for i in range(self.dims):
				self._population[i][observed[i]] += 1
				self._population[i][stated_ideal[i]] += 1
			self._loss_cache = self._r2_variance_cache = None
		state = self._partner(date.partner)
		previous_proposals = state.proposals_to
		decision = self._date_decision(date, observed, stated_ideal)
		est = min(self._est.get(date.partner, 0.0), self._estimate(observed, stated_ideal))
		if est <= self._quality_bar(date):
			state.proposals_to = previous_proposals
			return Decision()
		return decision

	def _ad_cost(self, ad):
		# Legacy diagnostic ranking; the submitted policy uses _ad_utilities.
		mine = self._gap(ad.own_dim, self.ideal[ad.own_dim], ad.own_value)
		theirs = self._gap(ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim])
		return mine + theirs / 2

	def _partner(self, index: int) -> _PartnerState:
		if index not in self._partners:
			self._partners[index] = _PartnerState()
		return self._partners[index]

	def _dim_means(self) -> Profile:
		if self._dim_count == 0:
			return tuple(5 for _ in range(self.dims))
		return tuple(min(MAX_VALUE, max(0, int(round(s / self._dim_count)))) for s in self._dim_sum)

	def _observe_dims(self, observed: Profile) -> None:
		for i, v in enumerate(observed):
			self._dim_sum[i] += v
		self._dim_count += 1

	@staticmethod
	def _percentile(values: list[float], q: float) -> float:
		s = sorted(values)
		if not s:
			return 0.0
		k = (len(s) - 1) * q
		lo = math.floor(k)
		hi = math.ceil(k)
		if lo == hi:
			return s[int(k)]
		return s[lo] * (hi - k) + s[hi] * (k - lo)

	def _gap(self, dim: int, want: int, have: int) -> int:
		gap = want - have
		if dim < self.monotonic and gap <= 0:
			return 0
		return gap * gap

	def _mirrored_ideal(self, stated_ideal: Profile, our_projection: Profile) -> bool:
		return all(a == b for a, b in zip(stated_ideal, our_projection, strict=True))

	def _adjust_profiles(
		self,
		partner: int,
		partner_ad: Advertisement,
		observed: Profile,
		stated_ideal: Profile,
		our_projection: Profile,
	) -> tuple[Profile, Profile, float]:
		state = self._partner(partner)
		trust = state.trust

		if state.observations and observed != state.observations[-1]:
			trust *= 0.65
		if not state.observations and abs(observed[partner_ad.own_dim] - partner_ad.own_value) > 1:
			trust *= 0.55

		if self._mirrored_ideal(stated_ideal, our_projection):
			trust *= 0.4
			means = self._dim_means()
			stated_ideal = means

		trust = max(0.15, min(1.0, trust))
		state.trust = trust

		means = self._dim_means()
		adj_obs = tuple(
			int(round(trust * o + (1.0 - trust) * means[i])) for i, o in enumerate(observed)
		)
		adj_ideal = tuple(
			int(round(trust * s + (1.0 - trust) * means[i])) for i, s in enumerate(stated_ideal)
		)
		return adj_obs, adj_ideal, trust

	def _estimate(
		self,
		observed: Profile,
		stated_ideal: Profile,
		trust: float = 1.0,
	) -> float:
		mine = decrement(self.ideal, observed, self.monotonic)
		theirs = decrement(stated_ideal, self.profile, self.monotonic)
		return self.bonus - mine - theirs / 2

	def on_outcome(self, outcome: DateOutcome) -> None:
		if outcome.partner_proposed:
			self._proposed_to_me.add(outcome.partner)
			ps = self._partner(outcome.partner)
			ps.proposals_to = 0
		if outcome.again:
			self._mutual_again[outcome.partner] = self._mutual_again.get(outcome.partner, 0) + 1
			ps = self._partner(outcome.partner)
			ps.dates_without_reciprocate = 0
		elif not outcome.married:
			ps = self._partner(outcome.partner)
			ps.dates_without_reciprocate += 1

	def _population_loss(self, dim, fixed, ideal_random):
		row = self._population[dim]
		return sum(
			w * (self._gap(dim, x, fixed) if ideal_random else self._gap(dim, fixed, x))
			for x, w in enumerate(row)
		) / sum(row)

	def _prior_losses(self):
		if self._loss_cache is None:
			self._loss_cache = (
				[self._population_loss(i, self.ideal[i], False) for i in range(self.dims)],
				[self._population_loss(i, self.profile[i], True) for i in range(self.dims)],
			)
		return self._loss_cache

	def _ad_utilities(self, ad):
		mine, theirs = self._prior_losses()
		a = (
			sum(mine)
			- mine[ad.own_dim]
			+ self._gap(ad.own_dim, self.ideal[ad.own_dim], ad.own_value)
		)
		b = (
			sum(theirs)
			- theirs[ad.seek_dim]
			+ self._gap(ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim])
		)
		return self.bonus - a - b / 2, self.bonus - b - a / 2

	def _r2_prior_variance(self):
		# Independent quadratic-loss variance. A constant marriage bonus shifts
		# utility but cannot change its variance. The population is public reports.
		if self._r2_variance_cache is None:
			total = 0.0
			for i, row in enumerate(self._population):
				n = sum(row)
				for fixed, reverse, weight in (
					(self.ideal[i], False, 1),
					(self.profile[i], True, 0.25),
				):
					losses = [
						self._gap(i, x, fixed) if reverse else self._gap(i, fixed, x)
						for x in range(11)
					]
					mean = sum(w * x for w, x in zip(row, losses)) / n  # noqa: B905 - equal lengths by construction
					total += weight * sum(w * (x - mean) ** 2 for w, x in zip(row, losses)) / n  # noqa: B905 - equal lengths by construction
			self._r2_variance_cache = total
		return self._r2_variance_cache


class Player3V1(Player3):
	"""Original Group 3 deliverable (ablation baseline)."""

	def project(self, date: DateContext) -> Profile:
		return self.profile

	def describe_ideal(self, date: DateContext, observed: Profile) -> Profile:
		return self.ideal

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		if not candidates:
			return []
		self._last_singles = turn.singles
		exploring = turn.round <= self._commit_round
		known = sorted(
			(c for c in candidates if self._est.get(c.index, 0.0) > 0),
			key=lambda c: self._est[c.index],
			reverse=True,
		)
		fresh = sorted(
			(c for c in candidates if c.index not in self._est),
			key=lambda c: (self._ad_cost(c.ad), c.index),
		)
		order = [c.index for c in known]
		order += [c.index for c in (fresh if exploring else fresh[:3])]
		seen: set[int] = set()
		chosen: list[int] = []
		for i in order:
			if i not in seen:
				seen.add(i)
				chosen.append(i)
		if self.list_limit is not None:
			chosen = chosen[: self.list_limit]
		return chosen

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		est = self._estimate(observed, stated_ideal)
		self._est[date.partner] = est
		self._seen.append(est)

		if est <= 0:
			return Decision(again=False, propose=False)

		bar = self._reservation_v1(date.rounds_left)
		willing = (
			date.partner in self._proposed_to_me or self._mutual_again.get(date.partner, 0) >= 2
		)
		propose = date.date_number >= self.MIN_DATES and est > 0 and (est > bar or willing)

		exploring = date.round <= self._commit_round
		pursue = est > max(bar, 0.5 * self.bonus) if exploring else est > 0.0
		return Decision(again=pursue or propose, propose=propose)

	def advertise(self):
		showable = range(self.monotonic) if self.monotonic else range(self.dims)
		own = max(showable, key=lambda i: self.profile[i])
		seek = max(range(self.dims), key=lambda i: abs(self.ideal[i] - 5))
		self._ad = Advertisement(own, min(MAX_VALUE, self.profile[own]), seek, self.ideal[seek])
		return self._ad


class Player3WithConfidence(Player3):
	"""Historical confidence ablation retained for existing course-test imports."""

	def _confidence_lb(self, est, trust):
		return est - (1.0 - trust) * 25.0

	def decide(self, date, observed, stated_ideal):
		return self._date_decision(date, observed, stated_ideal)

	def choose_dates(self, candidates, turn):
		if not candidates:
			return []
		self._last_singles = turn.singles
		known = sorted(
			(c for c in candidates if self._est.get(c.index, 0.0) > 0),
			key=lambda c: self._est[c.index],
			reverse=True,
		)
		fresh = sorted(
			(c for c in candidates if c.index not in self._est),
			key=lambda c: (self._ad_cost(c.ad), c.index),
		)
		exploring = turn.rounds_left > max(3, int(self.rounds * 0.35))
		order = list(dict.fromkeys(c.index for c in known + (fresh if exploring else fresh[:3])))
		if self.list_limit is not None:
			order = order[: self.list_limit]
		return [
			i
			for i in order
			if i in self._proposed_to_me
			or i not in self._partners
			or self._partners[i].proposals_to < 4
		][:8]
