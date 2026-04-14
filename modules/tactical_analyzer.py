"""
Tactical Analyzer - Professional Tactical Intelligence
======================================================

Generates sophisticated tactical analysis beyond basic statistics:
- Zone concentration patterns
- Passing chains and sequences
- Tactical narrative using Claude API
- Professional-grade commentary for match events

Author: TacticEYE2 Team
Date: 2026-04-14
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ZonePosition(Enum):
    """Zone positioning reference"""
    DEFENSIVE = "defensive"
    MIDFIELD = "midfield"
    OFFENSIVE = "offensive"


@dataclass
class PassingChain:
    """Represents a chain of consecutive passes"""
    team_id: int
    passes: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    zones: List[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        """Number of passes in chain"""
        return len(self.passes)

    @property
    def duration(self) -> float:
        """Duration in seconds"""
        return self.end_time - self.start_time if self.end_time > 0 else 0.0

    @property
    def is_active(self) -> bool:
        """Whether this chain is still ongoing"""
        return self.end_time == 0.0

    def to_dict(self) -> Dict:
        """Serialize chain"""
        return {
            'team_id': self.team_id,
            'length': self.length,
            'duration': self.duration,
            'zones': self.zones,
            'is_active': self.is_active,
            'passes': self.passes
        }


@dataclass
class ZoneDominance:
    """Zone dominance pattern"""
    team_id: int
    zones: List[str]
    concentration: float  # % of possession in these zones
    duration: float  # Seconds of focused play
    timestamp: float = field(default_factory=time.time)


class ZoneFocusAnalyzer:
    """
    Analyzes which zones teams concentrate their play in.

    Tracks:
    - Zone possession over time
    - Dominant zone patterns
    - Zone transitions (tactical shifts)
    """

    def __init__(self, window_size: int = 10, concentration_threshold: float = 0.40):
        """
        Args:
            window_size: Number of measurements to track (for trending)
            concentration_threshold: % threshold to consider zone dominant (0-1)
        """
        self.window_size = window_size
        self.concentration_threshold = concentration_threshold

        # History of zone measurements
        self.zone_history: Dict[int, deque] = {
            0: deque(maxlen=window_size),  # Team 0
            1: deque(maxlen=window_size)   # Team 1
        }

        # Zone name mapping (will be updated with actual zone names from model)
        self.zone_names: Dict[int, str] = {}

    def set_zone_names(self, zone_names: Dict[int, str]):
        """Update zone name mapping"""
        self.zone_names = zone_names

    def analyze(self, zone_stats: Dict[int, List[int]],
                zone_names: Optional[Dict[int, str]] = None) -> Dict[int, Dict]:
        """
        Analyze zone possession patterns.

        Args:
            zone_stats: {team_id: [zone_frames_0, zone_frames_1, ...]}
            zone_names: {zone_id: 'zone_name'} mapping

        Returns:
            {team_id: {'dominant_zones': [str], 'concentration': float, 'pattern': str}}
        """
        if zone_names:
            self.set_zone_names(zone_names)

        results = {}

        for team_id in [0, 1]:
            if team_id not in zone_stats:
                continue

            zone_frames = zone_stats[team_id]
            if not zone_frames or sum(zone_frames) == 0:
                continue

            # Calculate zone percentages
            total_frames = sum(zone_frames)
            zone_percentages = [(f / total_frames) for f in zone_frames]

            # Find top zones that make up the majority of possession
            # Sort zones by possession %
            zone_tuples = [(i, zone_percentages[i]) for i in range(len(zone_percentages))]
            zone_tuples.sort(key=lambda x: x[1], reverse=True)

            # Take top zones until we reach 60% concentration
            cumulative = 0.0
            dominant_zone_ids = []
            for zone_id, pct in zone_tuples:
                cumulative += pct
                if zone_id not in dominant_zone_ids:
                    dominant_zone_ids.append(zone_id)
                if cumulative >= 0.60:  # 60% threshold for overall concentration
                    break

            # Get zone names
            dominant_zone_names = [
                self.zone_names.get(i, f"Zone_{i}")
                for i in dominant_zone_ids
            ]

            # Calculate concentration (sum of top zones)
            concentration = sum(zone_percentages[i] for i in dominant_zone_ids)

            # Determine tactical pattern
            pattern = self._determine_pattern(dominant_zone_ids, concentration)

            # Store in history
            self.zone_history[team_id].append({
                'dominant_zones': dominant_zone_names,
                'concentration': concentration,
                'pattern': pattern
            })

            results[team_id] = {
                'dominant_zones': dominant_zone_names,
                'concentration': concentration,
                'pattern': pattern,
                'zone_details': [
                    {
                        'zone_name': self.zone_names.get(i, f"Zone_{i}"),
                        'percentage': zone_percentages[i] * 100
                    }
                    for i in range(len(zone_frames))
                ]
            }

        return results

    def detect_zone_shift(self, team_id: int, shift_threshold: float = 0.20) -> Optional[str]:
        """
        Detect if team is shifting tactical focus (changing zones).

        Args:
            team_id: Team to analyze
            shift_threshold: % change to trigger shift detection

        Returns:
            Description of shift or None
        """
        history = self.zone_history[team_id]
        if len(history) < 2:
            return None

        current = history[-1]
        previous = history[-2]

        current_zones = set(current['dominant_zones'])
        previous_zones = set(previous['dominant_zones'])

        # Check if zones changed significantly
        if current_zones != previous_zones:
            new_zones = current_zones - previous_zones
            lost_zones = previous_zones - current_zones

            if new_zones or lost_zones:
                msg_parts = []
                if new_zones:
                    msg_parts.append(f"focusing on {', '.join(new_zones)}")
                if lost_zones:
                    msg_parts.append(f"reducing play in {', '.join(lost_zones)}")

                return " and ".join(msg_parts) if msg_parts else None

        return None

    def _determine_pattern(self, zone_ids: List[int], concentration: float) -> str:
        """Determine tactical pattern from zones"""
        if not zone_ids:
            return "balanced"

        if concentration > 0.70:
            return "concentrated"
        elif concentration > 0.50:
            return "focused"
        else:
            return "distributed"


class PassingChainDetector:
    """
    Detects and tracks passing chains.

    Identifies sequences of 3+ consecutive passes by same team.
    Maintains active chains and completed chain history.
    """

    def __init__(self, min_chain_length: int = 3, event_window_size: int = 30):
        """
        Args:
            min_chain_length: Minimum passes to consider a chain (3+)
            event_window_size: Keep last N pass events in memory
        """
        self.min_chain_length = min_chain_length
        self.event_window_size = event_window_size

        # Recent pass events
        self.pass_events: deque = deque(maxlen=event_window_size)

        # Active chains by team
        self.active_chains: Dict[int, PassingChain] = {0: None, 1: None}

        # Completed chains for analysis
        self.completed_chains: List[PassingChain] = []

    def update(self, event: Dict[str, Any], fps: float = 30.0):
        """
        Update with new pass event.

        Args:
            event: Pass event dict with keys: type, team, from_player, to_player, frame, timestamp, zone
            fps: Frames per second (for timestamp calculation if not provided)
        """
        if event.get('type') != 'pass':
            return

        team_id = event.get('team')
        if team_id not in [0, 1]:
            return

        timestamp = event.get('timestamp', event.get('frame', 0) / fps)
        zone = event.get('zone', 'unknown')

        self.pass_events.append({
            'team': team_id,
            'from_player': event.get('from_player'),
            'to_player': event.get('to_player'),
            'timestamp': timestamp,
            'zone': zone
        })

        # Update active chain for this team
        if self.active_chains[team_id] is None:
            # Start new chain
            self.active_chains[team_id] = PassingChain(
                team_id=team_id,
                start_time=timestamp
            )

        chain = self.active_chains[team_id]
        chain.passes.append({
            'from': event.get('from_player'),
            'to': event.get('to_player'),
            'timestamp': timestamp
        })
        chain.end_time = timestamp
        if zone not in chain.zones:
            chain.zones.append(zone)

    def handle_possession_loss(self, team_id: int, timestamp: float):
        """
        Handle when team loses possession (ends the chain).

        Args:
            team_id: Team that lost possession
            timestamp: When possession was lost
        """
        if self.active_chains[team_id] is not None:
            chain = self.active_chains[team_id]

            if chain.length >= self.min_chain_length:
                self.completed_chains.append(chain)
                logger.info(f"Chain completed: Team {team_id}, length: {chain.length}, zones: {chain.zones}")

            self.active_chains[team_id] = None

    def get_notable_chains(self, team_id: int, min_length: int = 5) -> List[PassingChain]:
        """
        Get notable passing chains for team.

        Args:
            team_id: Team to analyze
            min_length: Minimum chain length to consider notable

        Returns:
            List of notable chains
        """
        notable = [c for c in self.completed_chains if c.team_id == team_id and c.length >= min_length]
        return sorted(notable, key=lambda c: c.length, reverse=True)

    def get_active_chain(self, team_id: int) -> Optional[PassingChain]:
        """Get current active chain (if any)"""
        chain = self.active_chains[team_id]
        if chain and chain.is_active and chain.length >= self.min_chain_length:
            return chain
        return None

    def get_chain_stats(self, team_id: int) -> Dict[str, Any]:
        """Get statistics about team's passing chains"""
        team_chains = [c for c in self.completed_chains if c.team_id == team_id]

        if not team_chains:
            return {
                'total_chains': 0,
                'avg_length': 0,
                'max_length': 0,
                'avg_duration': 0
            }

        lengths = [c.length for c in team_chains]
        durations = [c.duration for c in team_chains]

        return {
            'total_chains': len(team_chains),
            'avg_length': sum(lengths) / len(lengths),
            'max_length': max(lengths),
            'avg_duration': sum(durations) / len(durations) if durations else 0,
            'chains': [c.to_dict() for c in team_chains[-5:]]  # Last 5 chains
        }


class TacticalInsightGenerator:
    """
    Generates tactical insights combining zone and passing analysis.
    """

    def __init__(self):
        self.zone_analyzer = ZoneFocusAnalyzer()
        self.chain_detector = PassingChainDetector()

    def analyze(self, zone_stats: Dict[int, List[int]],
                events: List[Dict[str, Any]],
                possession_stats: Dict[str, Any],
                zone_names: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive tactical insights.

        Args:
            zone_stats: Zone possession data
            events: Recent pass events
            possession_stats: Current possession info (frames_by_team, passes_by_team, etc.)
            zone_names: Zone name mapping

        Returns:
            Dict with tactical insights for both teams
        """
        # Update zone analysis
        zone_analysis = self.zone_analyzer.analyze(zone_stats, zone_names)

        # Process events for chain detection
        for event in events:
            if event.get('type') == 'pass':
                self.chain_detector.update(event)
            elif event.get('type') == 'possession_loss':
                team_id = event.get('team')
                self.chain_detector.handle_possession_loss(team_id, event.get('timestamp', 0))

        # Compilation of insights
        insights = {
            'timestamp': time.time(),
            'teams': {}
        }

        for team_id in [0, 1]:
            team_insights = {}

            # Zone information
            if team_id in zone_analysis:
                team_insights['zones'] = zone_analysis[team_id]

            # Chain information
            notable_chains = self.chain_detector.get_notable_chains(team_id, min_length=4)
            active_chain = self.chain_detector.get_active_chain(team_id)

            team_insights['passing_chains'] = {
                'active': active_chain.to_dict() if active_chain else None,
                'notable_recent': [c.to_dict() for c in notable_chains[:3]],
                'stats': self.chain_detector.get_chain_stats(team_id)
            }

            # Tactical pattern summary
            team_insights['pattern'] = self._summarize_pattern(
                team_id, zone_analysis.get(team_id, {}),
                possession_stats, notable_chains
            )

            insights['teams'][team_id] = team_insights

        return insights

    def _summarize_pattern(self, team_id: int, zone_info: Dict,
                          possession_stats: Dict, chains: List[PassingChain]) -> str:
        """Summarize tactical pattern in human-readable form"""
        pattern_parts = []

        # Zone pattern
        if zone_info.get('dominant_zones'):
            zones = ', '.join(zone_info['dominant_zones'][:3])
            pattern_parts.append(f"Focusing on {zones}")

        # Passing pattern
        if chains:
            avg_length = sum(c.length for c in chains) / len(chains)
            if avg_length >= 5:
                pattern_parts.append("Building effective play sequences")
            else:
                pattern_parts.append("Playing short passing style")

        # Possession pattern
        pct = possession_stats.get('possession_percent', {}).get(team_id, 0)
        if pct >= 60:
            pattern_parts.append("Dominating possession")
        elif pct <= 40:
            pattern_parts.append("Counter-attacking focus")

        return "; ".join(pattern_parts) if pattern_parts else "Balanced style"


class ClaudeNarrativeGenerator:
    """
    Generates professional narrative commentary using Claude API.

    Calls Claude API to create ESPN-style match analysis based on game context.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.client = None
        self.last_call_time = 0.0
        self.min_call_interval = 60.0  # Max 1 call per 60 seconds
        self.call_count = 0

        if self.api_key:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
                logger.info("✓ Claude API initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Claude API: {e}")
                logger.warning("Narrative generation will be disabled")

    def should_generate(self) -> bool:
        """Check if enough time has passed since last API call"""
        elapsed = time.time() - self.last_call_time
        return elapsed >= self.min_call_interval

    async def generate_narrative(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Generate professional narrative for current match moment.

        Args:
            context: Dict with keys:
                - possession: {0: float, 1: float} % by team
                - zones: Zone analysis dict
                - chains: Passing chain info
                - event_type: str (e.g., 'zone_shift', 'chain_detection')
                - time_minutes: int

        Returns:
            Professional commentary string or None if API unavailable
        """
        if not self.client or not self.should_generate():
            return None

        try:
            # Build prompt with game context
            prompt = self._build_prompt(context)

            # Call Claude API
            self.last_call_time = time.time()
            self.call_count += 1

            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                system=self._get_system_prompt(),
                messages=[{"role": "user", "content": prompt}]
            )

            narrative = message.content[0].text if message.content else None

            if narrative:
                logger.info(f"Generated narrative (call #{self.call_count}): {narrative[:80]}...")

            return narrative

        except Exception as e:
            logger.error(f"Error generating narrative: {e}")
            return None

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for Claude"""
        lines = [
            "Current match moment analysis:",
            ""
        ]

        # Match time
        if 'time_minutes' in context:
            lines.append(f"Match time: Minute {context['time_minutes']}")

        # Possession
        if 'possession' in context:
            pct = context['possession']
            lines.append(f"Possession: Team A {pct.get(0, 0):.1f}% | Team B {pct.get(1, 0):.1f}%")

        # Zones
        zones = context.get('zones', {})
        for team_id in [0, 1]:
            if team_id in zones and zones[team_id].get('dominant_zones'):
                zone_str = ', '.join(zones[team_id]['dominant_zones'])
                lines.append(f"Team {team_id + 1} focusing on: {zone_str}")

        # Chains
        chains = context.get('chains', {})
        for team_id in [0, 1]:
            if team_id in chains:
                active = chains[team_id].get('active')
                if active and active.get('length', 0) >= 5:
                    lines.append(f"Team {team_id + 1} has a {active['length']}-pass sequence in progress")

        # Event type
        if context.get('event_type'):
            lines.append(f"Notable moment: {context['event_type']}")

        lines.append("\nProvide a 1-2 sentence professional match commentary about this moment.")

        return "\n".join(lines)

    def _get_system_prompt(self) -> str:
        """Get system prompt for Claude"""
        return """You are an expert football (soccer) match analyst with the style of ESPN/Sky Sports commentators.
Provide brief, insightful professional commentary on match moments.
Focus on tactical observations and game dynamics.
Be concise (1-2 sentences max) and use clear language.
Avoid speculation; stick to observable data."""


# ============================================================================
# Main Tactical Analyzer (Facade)
# ============================================================================

class TacticalAnalyzer:
    """
    Main interface for tactical analysis.

    Combines all analysis components and provides single entry point.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.insight_generator = TacticalInsightGenerator()
        self.narrative_generator = ClaudeNarrativeGenerator()
        self.last_insights = None
        self.zone_names = {}

    def set_zone_names(self, zone_names: Dict[int, str]):
        """Configure zone names for display"""
        self.zone_names = zone_names
        self.insight_generator.zone_analyzer.set_zone_names(zone_names)

    def analyze(self, spatial_stats: Dict[str, Any],
                possession_stats: Dict[str, Any],
                events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Perform complete tactical analysis.

        Args:
            spatial_stats: Zone possession data from tracker
            possession_stats: Possession statistics
            events: Recent game events (passes, etc.)

        Returns:
            Dict with tactical insights and optional narrative
        """
        events = events or []

        # Get zone stats
        zone_stats = spatial_stats.get('possession_by_zone', {})

        # Generate insights
        insights = self.insight_generator.analyze(
            zone_stats, events, possession_stats,
            self.zone_names
        )

        self.last_insights = insights
        return insights

    async def get_narrative(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Get professional commentary if appropriate.

        Args:
            context: Context dict with possession, zones, chains, etc.

        Returns:
            Professional narrative or None
        """
        return await self.narrative_generator.generate_narrative(context)
