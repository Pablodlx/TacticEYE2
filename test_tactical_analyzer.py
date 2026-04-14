#!/usr/bin/env python3
"""
Test script for tactical analyzer integration
"""

import sys
sys.path.insert(0, '/home/pablodlx/TacticEYE2_github')

import logging
logging.basicConfig(level=logging.INFO)

from modules.tactical_analyzer import TacticalAnalyzer, ZoneFocusAnalyzer, PassingChainDetector
from modules.match_alert_system import MatchAlertSystem

def test_zone_focus_analyzer():
    """Test ZoneFocusAnalyzer"""
    print("\n=== Testing ZoneFocusAnalyzer ===")

    analyzer = ZoneFocusAnalyzer()

    # Set zone names
    zone_names = {
        0: 'Defensive Left',
        1: 'Defensive Center',
        2: 'Defensive Right',
        3: 'Midfield Left',
        4: 'Midfield Center',
        5: 'Midfield Right',
        6: 'Offensive Left',
        7: 'Offensive Center',
        8: 'Offensive Right'
    }
    analyzer.set_zone_names(zone_names)

    # Test data: Team 0 concentrated in attacking zones (6,7,8)
    zone_stats = {
        0: [10, 10, 10, 20, 20, 20, 100, 120, 110],  # Team 0: concentrated offensive
        1: [100, 120, 110, 20, 20, 20, 10, 10, 10]   # Team 1: concentrated defensive
    }

    result = analyzer.analyze(zone_stats, zone_names)

    for team_id, analysis in result.items():
        print(f"\nTeam {team_id}:")
        print(f"  Dominant zones: {analysis['dominant_zones']}")
        print(f"  Concentration: {analysis['concentration']:.1%}")
        print(f"  Pattern: {analysis['pattern']}")

    return True

def test_passing_chain_detector():
    """Test PassingChainDetector"""
    print("\n=== Testing PassingChainDetector ===")

    detector = PassingChainDetector(min_chain_length=3)

    # Simulate a 5-pass sequence
    fps = 30.0
    events = [
        {'type': 'pass', 'team': 0, 'from_player': 1, 'to_player': 2, 'frame': 0, 'timestamp': 0, 'zone': 'Offensive Center'},
        {'type': 'pass', 'team': 0, 'from_player': 2, 'to_player': 3, 'frame': 5, 'timestamp': 5/fps, 'zone': 'Offensive Center'},
        {'type': 'pass', 'team': 0, 'from_player': 3, 'to_player': 4, 'frame': 10, 'timestamp': 10/fps, 'zone': 'Offensive Right'},
        {'type': 'pass', 'team': 0, 'from_player': 4, 'to_player': 5, 'frame': 15, 'timestamp': 15/fps, 'zone': 'Offensive Right'},
        {'type': 'pass', 'team': 0, 'from_player': 5, 'to_player': 1, 'frame': 20, 'timestamp': 20/fps, 'zone': 'Midfield Center'},
    ]

    for event in events:
        detector.update(event, fps)

    # End the chain
    detector.handle_possession_loss(0, 25/fps)

    # Get stats
    stats = detector.get_chain_stats(0)
    print(f"\nTeam 0 chain stats:")
    print(f"  Total chains: {stats['total_chains']}")
    print(f"  Max length: {stats['max_length']}")
    print(f"  Avg duration: {stats['avg_duration']:.2f}s")

    if stats['total_chains'] > 0:
        print(f"\nSample chain:")
        chain = stats['chains'][0]
        print(f"  Length: {chain['length']}")
        print(f"  Duration: {chain['duration']:.2f}s")
        print(f"  Zones: {chain['zones']}")

    return stats['total_chains'] > 0 and stats['max_length'] == 5

def test_tactical_analyzer():
    """Test TacticalAnalyzer"""
    print("\n=== Testing TacticalAnalyzer ===")

    analyzer = TacticalAnalyzer(fps=30.0)

    zone_names = {i: f'Zone_{i}' for i in range(9)}
    analyzer.set_zone_names(zone_names)

    spatial_stats = {
        'possession_by_zone': {
            0: [10, 10, 10, 20, 20, 20, 100, 120, 110],
            1: [100, 120, 110, 20, 20, 20, 10, 10, 10]
        }
    }

    possession_stats = {
        'possession_percent': {0: 55, 1: 45},
        'passes_by_team': {0: 150, 1: 80}
    }

    insights = analyzer.analyze(spatial_stats, possession_stats, events=[])

    print("\nTactical insights generated:")
    for team_id in [0, 1]:
        team_data = insights['teams'].get(team_id, {})
        if team_data.get('zones'):
            print(f"\nTeam {team_id}:")
            print(f"  Zones: {team_data['zones']}")
            print(f"  Pattern: {team_data.get('pattern', 'N/A')}")

    return True

def test_alert_system():
    """Test MatchAlertSystem with tactical analysis"""
    print("\n=== Testing MatchAlertSystem ===")

    # Use 60 second check interval = 1800 frames at 30fps
    alert_system = MatchAlertSystem(fps=30.0, check_interval_seconds=60.0, min_alert_interval_seconds=5.0)

    # Test data with strong zone concentration
    possession_stats = {
        'frames_by_team': {0: 1500, 1: 1000},  # 60% vs 40%
        'passes_by_team': {0: 150, 1: 80},
        'current_team': 0,
        'possession_changes': 5,
        'possession_percent': {0: 60, 1: 40}
    }

    # Zone data: Team 0 concentrated in attacking zones (6,7,8)
    spatial_stats = {
        'possession_by_zone': {
            0: [20, 20, 20, 40, 40, 40, 300, 350, 325],  # Team 0: heavily attacking
            1: [350, 325, 300, 40, 40, 40, 20, 20, 20]   # Team 1: heavily defending
        }
    }

    # First call won't trigger (frame 0 < check_interval_frames = 1800)
    alerts1 = alert_system.analyze_and_generate_alerts(
        frame_id=1800,  # Exactly at first check point
        possession_stats=possession_stats,
        spatial_stats=spatial_stats
    )

    print(f"\nGenerated {len(alerts1)} alerts at frame 1800")
    for alert in alerts1:
        print(f"  [{alert.type}] {alert.title}")
        if alert.data.get('dominant_zones'):
            print(f"    → Zones: {alert.data['dominant_zones']}")

    return len(alerts1) > 0

if __name__ == '__main__':
    print("=" * 60)
    print("Tactical Analyzer Integration Tests")
    print("=" * 60)

    results = []

    try:
        results.append(("Zone Focus Analyzer", test_zone_focus_analyzer()))
    except Exception as e:
        print(f"❌ Zone Focus Analyzer: {e}")
        results.append(("Zone Focus Analyzer", False))

    try:
        results.append(("Passing Chain Detector", test_passing_chain_detector()))
    except Exception as e:
        print(f"❌ Passing Chain Detector: {e}")
        results.append(("Passing Chain Detector", False))

    try:
        results.append(("Tactical Analyzer", test_tactical_analyzer()))
    except Exception as e:
        print(f"❌ Tactical Analyzer: {e}")
        results.append(("Tactical Analyzer", False))

    try:
        results.append(("Alert System", test_alert_system()))
    except Exception as e:
        print(f"❌ Alert System: {e}")
        results.append(("Alert System", False))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(r[1] for r in results)
    exit_code = 0 if all_passed else 1

    print("\n" + ("All tests passed! ✓" if all_passed else "Some tests failed ✗"))
    sys.exit(exit_code)
