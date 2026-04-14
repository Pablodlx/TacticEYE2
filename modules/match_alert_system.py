"""
Match Alert System - Sistema Inteligente de Alertas Tácticas
=============================================================

Analiza estadísticas del partido en tiempo real y genera alertas contextuales
sobre patrones tácticos, posesión, pases, y anomalías del juego.

Versión mejorada con análisis táctico profesional usando Claude API.

Author: TacticEYE2 Team
Date: 2026-04-14
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
import asyncio
import logging

try:
    from modules.tactical_analyzer import TacticalAnalyzer
except ImportError:
    TacticalAnalyzer = None

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Representa una alerta táctica"""
    id: str
    timestamp: float
    frame_id: int
    type: str  # 'possession', 'passing', 'zone', 'tactical', 'warning', 'zone_concentration', 'passing_chain', 'tactical_shift', 'tactical_excellence'
    severity: str  # 'info', 'warning', 'critical'
    title: str
    message: str
    team_id: Optional[int] = None
    data: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'frame_id': self.frame_id,
            'type': self.type,
            'severity': self.severity,
            'title': self.title,
            'message': self.message,
            'team_id': self.team_id,
            'data': self.data
        }


class MatchAlertSystem:
    """
    Sistema de alertas que monitorea estadísticas y genera notificaciones inteligentes.
    
    Detecta:
    - Acumulación de posesión en zonas específicas
    - Periodos sin completar pases
    - Cambios bruscos de posesión
    - Dominancia extendida de un equipo
    - Anomalías tácticas
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        check_interval_seconds: float = 60.0,  # Chequear cada minuto
        min_alert_interval_seconds: float = 10.0  # Permitir alertas cada 10 segundos
    ):
        """
        Args:
            fps: Frames por segundo del video
            check_interval_seconds: Intervalo de tiempo para evaluar estadísticas
            min_alert_interval_seconds: Tiempo mínimo entre alertas del mismo tipo
        """
        self.fps = fps
        self.check_interval_frames = int(check_interval_seconds * fps)
        self.min_alert_interval = min_alert_interval_seconds

        # Estado interno
        self.last_check_frame: int = 0
        self.alert_counter: int = 0
        self.last_alert_time: Dict[str, float] = {}

        # Histórico de estadísticas para análisis temporal
        self.possession_history: List[Tuple[int, Dict[int, float]]] = []
        self.passes_history: List[Tuple[int, Dict[int, int]]] = []
        self.zone_possession_history: List[Tuple[int, Dict]] = []

        # Thresholds configurables (más sensibles para alertas frecuentes)
        self.POSSESSION_DOMINANCE_THRESHOLD = 60.0  # % posesión para considerar dominio
        self.LONG_NO_PASS_THRESHOLD_SECONDS = 30.0  # Segundos sin pases
        self.ZONE_PRESSURE_THRESHOLD = 0.55  # 55% posesión rival en zona defensiva
        self.POSSESSION_SWING_THRESHOLD = 15.0  # Cambio de % para detectar giro de partido
        self.SUMMARY_INTERVAL_SECONDS = 90.0  # Resumen cada 90 segundos
        self.last_summary_time = 0.0

        # === TACTICAL ANALYZER ===
        if TacticalAnalyzer is not None:
            self.tactical_analyzer = TacticalAnalyzer(fps=fps)
            logger.info("✓ Tactical analyzer initialized")
        else:
            self.tactical_analyzer = None
            logger.warning("⚠ Tactical analyzer not available (modules.tactical_analyzer import failed)")

        # Event history for tactical analysis
        self.event_history: List[Dict] = []
        self.max_event_history = 50

        # Zone shift detection
        self.last_zone_analysis: Dict[int, Dict] = {}
        
    def should_check(self, frame_id: int) -> bool:
        """Determina si es momento de evaluar y potencialmente generar alertas"""
        return (frame_id - self.last_check_frame) >= self.check_interval_frames
    
    def can_send_alert(self, alert_type: str) -> bool:
        """Verifica si ha pasado suficiente tiempo desde la última alerta del mismo tipo"""
        last_time = self.last_alert_time.get(alert_type, 0)
        return (time.time() - last_time) >= self.min_alert_interval
    
    def _mark_alert_sent(self, alert_type: str):
        """Marca una alerta como enviada"""
        self.last_alert_time[alert_type] = time.time()
    
    def _create_alert(
        self,
        frame_id: int,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        team_id: Optional[int] = None,
        data: Optional[Dict] = None
    ) -> Alert:
        """Crea una nueva alerta con ID único"""
        self.alert_counter += 1
        alert_id = f"alert_{self.alert_counter}_{int(time.time())}"
        
        return Alert(
            id=alert_id,
            timestamp=time.time(),
            frame_id=frame_id,
            type=alert_type,
            severity=severity,
            title=title,
            message=message,
            team_id=team_id,
            data=data or {}
        )
    
    def analyze_and_generate_alerts(
        self,
        frame_id: int,
        possession_stats: Dict,
        spatial_stats: Optional[Dict] = None
    ) -> List[Alert]:
        """
        Analiza estadísticas actuales y genera alertas si se detectan patrones relevantes.
        
        Args:
            frame_id: Frame actual
            possession_stats: Estadísticas de posesión
                {
                    'frames_by_team': {0: int, 1: int},
                    'passes_by_team': {0: int, 1: int},
                    'current_team': int,
                    'possession_changes': int
                }
            spatial_stats: Estadísticas espaciales (opcional)
                {
                    'zone_possession': {zone_id: {'team_0': float, 'team_1': float}},
                    'heatmap_data': {...}
                }
        
        Returns:
            Lista de alertas generadas
        """
        if not self.should_check(frame_id):
            return []
        
        self.last_check_frame = frame_id
        alerts = []
        
        # Calcular estadísticas actuales
        frames_by_team = possession_stats.get('frames_by_team', {})
        passes_by_team = possession_stats.get('passes_by_team', {})
        
        total_frames = sum(frames_by_team.values())
        if total_frames == 0:
            return []
        
        # Calcular porcentajes de posesión
        possession_percent = {}
        for team_id, frames in frames_by_team.items():
            possession_percent[team_id] = (frames / total_frames) * 100
        
        # Almacenar en histórico
        self.possession_history.append((frame_id, possession_percent.copy()))
        self.passes_history.append((frame_id, passes_by_team.copy()))
        
        # === ALERTA 1: Dominio de posesión ===
        alerts.extend(self._check_possession_dominance(frame_id, possession_percent))
        
        # === ALERTA 2: Falta de pases ===
        alerts.extend(self._check_passing_drought(frame_id, passes_by_team, total_frames))
        
        # === ALERTA 3: Cambio de momentum ===
        alerts.extend(self._check_possession_swing(frame_id, possession_percent))
        
        # === ALERTA 4: Estadísticas de juego ===
        alerts.extend(self._check_possession_changes(frame_id, possession_stats))
        
        # === ALERTA 5: Análisis espacial (si disponible) ===
        if spatial_stats:
            alerts.extend(self._check_spatial_pressure(frame_id, spatial_stats, possession_percent))

        # === ALERTA 6: ANÁLISIS TÁCTICO AVANZADO ===
        # Detección de patrones tácticos usando TacticalAnalyzer
        if spatial_stats and self.tactical_analyzer:
            # Agregar estadísticas de posesión al spatial_stats si no están presentes
            if 'possession_percent' not in spatial_stats:
                spatial_stats['possession_percent'] = possession_percent

            # Extraer eventos si están disponibles en spatial_stats
            events = spatial_stats.get('recent_events', [])

            # Análisis zonal
            alerts.extend(self._check_zone_dominance_patterns(frame_id, spatial_stats))

            # Análisis de cadenas de pases
            if events:
                alerts.extend(self._check_passing_chain_efficiency(frame_id, events))

            # Detección de cambios tácticos
            alerts.extend(self._check_tactical_shift_detection(frame_id))

            # Generar alerta profesional con análisis (solo en momentos clave)
            professional_alert = self._generate_professional_alert(frame_id, possession_stats, spatial_stats)
            if professional_alert:
                alerts.append(professional_alert)

        # === ALERTA 7: Resumen periódico del partido (SIEMPRE) ===
        # Esta alerta se genera independientemente del intervalo de chequeo
        summary = self._check_periodic_summary(frame_id, possession_stats, possession_percent)
        alerts.extend(summary)
        
        return alerts
    
    def _check_possession_dominance(self, frame_id: int, possession_percent: Dict) -> List[Alert]:
        """Detecta si un equipo tiene dominio claro del partido"""
        alerts = []
        alert_type = "possession_dominance"
        
        if not self.can_send_alert(alert_type):
            return alerts
        
        for team_id, pct in possession_percent.items():
            if pct >= self.POSSESSION_DOMINANCE_THRESHOLD:
                opponent_id = 1 - team_id
                opponent_pct = possession_percent.get(opponent_id, 0)
                
                alert = self._create_alert(
                    frame_id=frame_id,
                    alert_type="possession",
                    severity="warning" if pct >= 75 else "info",
                    title=f"⚠️ Dominio del Equipo {team_id}",
                    message=f"El Equipo {team_id} domina claramente la posesión con {pct:.1f}% vs {opponent_pct:.1f}% del rival. Considera ajustar la presión.",
                    team_id=team_id,
                    data={
                        'possession_team': pct,
                        'possession_opponent': opponent_pct,
                        'difference': pct - opponent_pct
                    }
                )
                alerts.append(alert)
                self._mark_alert_sent(alert_type)
                break
        
        return alerts
    
    def _check_passing_drought(self, frame_id: int, passes_by_team: Dict, total_frames: int) -> List[Alert]:
        """Detecta si un equipo lleva mucho tiempo sin completar pases"""
        alerts = []
        
        # Buscar último segmento de pases en el histórico
        if len(self.passes_history) < 2:
            return alerts
        
        # Comparar pases actuales con hace N segundos
        lookback_frames = int(self.LONG_NO_PASS_THRESHOLD_SECONDS * self.fps)
        
        for team_id in [0, 1]:
            current_passes = passes_by_team.get(team_id, 0)
            
            # Buscar pases hace N segundos
            target_frame = frame_id - lookback_frames
            past_passes = None
            
            for hist_frame, hist_passes in reversed(self.passes_history):
                if hist_frame <= target_frame:
                    past_passes = hist_passes.get(team_id, 0)
                    break
            
            if past_passes is not None:
                passes_in_period = current_passes - past_passes
                
                # Si no ha completado pases en el periodo
                alert_type = f"no_passes_team{team_id}"
                if passes_in_period < 4 and self.can_send_alert(alert_type):
                    time_seconds = self.LONG_NO_PASS_THRESHOLD_SECONDS
                    
                    alert = self._create_alert(
                        frame_id=frame_id,
                        alert_type="passing",
                        severity="warning",
                        title=f"🔴 Equipo {team_id} sin pases fluidos",
                        message=f"El Equipo {team_id} lleva {time_seconds:.0f} segundos con menos de 4 pases completados. Perdiendo control del balón.",
                        team_id=team_id,
                        data={
                            'passes_in_period': passes_in_period,
                            'period_seconds': time_seconds
                        }
                    )
                    alerts.append(alert)
                    self._mark_alert_sent(alert_type)
        
        return alerts
    
    def _check_possession_swing(self, frame_id: int, possession_percent: Dict) -> List[Alert]:
        """Detecta cambios bruscos en el control del partido"""
        alerts = []
        
        # Necesitamos al menos 2 mediciones con suficiente separación
        if len(self.possession_history) < 2:
            return alerts
        
        # Comparar con medición anterior (hace ~30 segundos)
        prev_frame, prev_possession = self.possession_history[-2]
        
        alert_type = "possession_swing"
        if not self.can_send_alert(alert_type):
            return alerts
        
        for team_id in [0, 1]:
            current_pct = possession_percent.get(team_id, 0)
            prev_pct = prev_possession.get(team_id, 0)
            swing = current_pct - prev_pct
            
            if abs(swing) >= self.POSSESSION_SWING_THRESHOLD:
                direction = "recuperado" if swing > 0 else "perdido"
                emoji = "📈" if swing > 0 else "📉"
                
                alert = self._create_alert(
                    frame_id=frame_id,
                    alert_type="tactical",
                    severity="info",
                    title=f"{emoji} Cambio de momentum - Equipo {team_id}",
                    message=f"El Equipo {team_id} ha {direction} {abs(swing):.1f}% de posesión en el último minuto. El partido está cambiando.",
                    team_id=team_id,
                    data={
                        'swing': swing,
                        'current_possession': current_pct,
                        'previous_possession': prev_pct
                    }
                )
                alerts.append(alert)
                self._mark_alert_sent(alert_type)
                break
        
        return alerts
    
    def _check_possession_changes(self, frame_id: int, possession_stats: Dict) -> List[Alert]:
        """Analiza el número de cambios de posesión para detectar juego caótico o controlado"""
        alerts = []
        
        possession_changes = possession_stats.get('possession_changes', 0)
        total_frames = sum(possession_stats.get('frames_by_team', {}).values())
        
        if total_frames == 0:
            return alerts
        
        # Calcular cambios por minuto
        total_minutes = (total_frames / self.fps) / 60
        if total_minutes < 1.0:
            return alerts
        
        changes_per_minute = possession_changes / total_minutes
        
        alert_type = "possession_intensity"
        
        # Juego muy fragmentado
        if changes_per_minute > 15 and self.can_send_alert(alert_type):
            alert = self._create_alert(
                frame_id=frame_id,
                alert_type="tactical",
                severity="info",
                title="⚡ Ritmo de juego intenso",
                message=f"Se registran {changes_per_minute:.1f} cambios de posesión por minuto. El juego es muy dinámico y fragmentado.",
                data={
                    'changes_per_minute': changes_per_minute,
                    'total_changes': possession_changes
                }
            )
            alerts.append(alert)
            self._mark_alert_sent(alert_type)
        # Juego muy controlado
        elif changes_per_minute < 4 and self.can_send_alert(alert_type):
            alert = self._create_alert(
                frame_id=frame_id,
                alert_type="tactical",
                severity="info",
                title="🎯 Juego controlado",
                message=f"Solo {changes_per_minute:.1f} cambios de posesión por minuto. Un equipo está controlando claramente el ritmo.",
                data={
                    'changes_per_minute': changes_per_minute,
                    'total_changes': possession_changes
                }
            )
            alerts.append(alert)
            self._mark_alert_sent(alert_type)
        
        return alerts
    
    def _check_spatial_pressure(self, frame_id: int, spatial_stats: Dict, possession_percent: Dict) -> List[Alert]:
        """Analiza presión en zonas específicas del campo"""
        alerts = []
        
        zone_possession = spatial_stats.get('zone_possession', {})
        if not zone_possession:
            return alerts
        
        # Analizar zonas defensivas (tercios)
        # Zona 0 = tercio defensivo equipo 0
        # Zona 2 = tercio defensivo equipo 1
        
        defensive_zones = {
            0: [0, 3, 6, 9],  # Tercio izquierdo equipo 0 (columnas 0)
            1: [2, 5, 8, 11]  # Tercio derecho equipo 1 (columnas 2)
        }
        
        for team_id, zones in defensive_zones.items():
            opponent_id = 1 - team_id
            
            # Calcular posesión del rival en zona defensiva
            opponent_possession_in_defense = 0
            total_in_defense = 0
            
            for zone_id in zones:
                if zone_id in zone_possession:
                    zone_data = zone_possession[zone_id]
                    opponent_frames = zone_data.get(f'team_{opponent_id}', 0)
                    team_frames = zone_data.get(f'team_{team_id}', 0)
                    
                    opponent_possession_in_defense += opponent_frames
                    total_in_defense += (opponent_frames + team_frames)
            
            if total_in_defense == 0:
                continue
            
            opponent_pct_in_defense = opponent_possession_in_defense / total_in_defense
            
            alert_type = f"zone_pressure_team{team_id}"
            
            # Alerta si el rival tiene mucha posesión en nuestra zona defensiva
            if opponent_pct_in_defense >= self.ZONE_PRESSURE_THRESHOLD and self.can_send_alert(alert_type):
                alert = self._create_alert(
                    frame_id=frame_id,
                    alert_type="zone",
                    severity="warning",
                    title=f"🛡️ Presión rival en zona defensiva - Equipo {team_id}",
                    message=f"El Equipo {opponent_id} acumula {opponent_pct_in_defense*100:.0f}% de posesión en tu tercio defensivo. Aumenta la salida de balón.",
                    team_id=team_id,
                    data={
                        'opponent_possession_in_defense': opponent_pct_in_defense * 100,
                        'zones_affected': len(zones)
                    }
                )
                alerts.append(alert)
                self._mark_alert_sent(alert_type)
        
        return alerts
    
    def _check_periodic_summary(self, frame_id: int, possession_stats: Dict, possession_percent: Dict) -> List[Alert]:
        """Genera resumen periódico del estado del partido"""
        alerts = []
        
        current_time = time.time()
        # El resumen se genera SIEMPRE cada 90 segundos, independiente de otros intervalos
        if (current_time - self.last_summary_time) < self.SUMMARY_INTERVAL_SECONDS:
            return alerts
        
        self.last_summary_time = current_time
        
        # Calcular tiempo de juego
        total_seconds = frame_id / self.fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        
        # No generar resumen antes del primer minuto
        if minutes < 1:
            return alerts
        
        # Obtener estadísticas
        passes_by_team = possession_stats.get('passes_by_team', {})
        
        # Determinar equipo dominante
        team_0_poss = possession_percent.get(0, 0)
        team_1_poss = possession_percent.get(1, 0)
        
        if abs(team_0_poss - team_1_poss) < 10:
            possession_status = "Posesión equilibrada"
            dominant_team = None
        elif team_0_poss > team_1_poss:
            possession_status = f"Equipo 0 domina"
            dominant_team = 0
        else:
            possession_status = f"Equipo 1 domina"
            dominant_team = 1
        
        # Crear mensaje de resumen
        message_parts = [
            f"⏱️ Minuto {minutes}:{seconds:02d}",
            f"\n📊 {possession_status}: {team_0_poss:.1f}% vs {team_1_poss:.1f}%",
            f"\n⚽ Pases: Equipo 0 ({passes_by_team.get(0, 0)}) vs Equipo 1 ({passes_by_team.get(1, 0)})"
        ]
        
        # Añadir observaciones
        if dominant_team is not None:
            dominant_poss = possession_percent.get(dominant_team, 0)
            if dominant_poss > 65:
                message_parts.append(f"\n⚠️ Control claro del Equipo {dominant_team}")
        
        # Analizar pases
        team_0_passes = passes_by_team.get(0, 0)
        team_1_passes = passes_by_team.get(1, 0)
        if team_0_passes > 0 and team_1_passes > 0:
            pass_ratio = team_0_passes / team_1_passes if team_1_passes > 0 else 0
            if pass_ratio > 2.0:
                message_parts.append(f"\n🎯 Equipo 0 con mejor circulación de balón")
            elif pass_ratio < 0.5:
                message_parts.append(f"\n🎯 Equipo 1 con mejor circulación de balón")
        
        alert = self._create_alert(
            frame_id=frame_id,
            alert_type="tactical",
            severity="info",
            title=f"📋 Resumen - Minuto {minutes}",
            message=''.join(message_parts),
            data={
                'time_minutes': minutes,
                'possession_percent': possession_percent,
                'passes': passes_by_team
            }
        )
        alerts.append(alert)
        
        return alerts

    # ============================================================================
    # TACTICAL ANALYSIS METHODS
    # ============================================================================

    def _check_zone_dominance_patterns(self, frame_id: int, spatial_stats: Dict) -> List[Alert]:
        """
        Detecta patrones de dominio zonal y concentración táctica.

        Genera alertas cuando un equipo concentra su juego en zonas específicas.
        """
        alerts = []

        if not self.tactical_analyzer or not spatial_stats:
            return alerts

        alert_type = "zone_concentration"

        try:
            # Obtener estadísticas de posesión por zona
            zone_stats = spatial_stats.get('possession_by_zone', {})
            if not zone_stats:
                return alerts

            # Analizar zonas para ambos equipos
            zone_analysis = self.tactical_analyzer.insight_generator.zone_analyzer.analyze(
                zone_stats, zone_names={}
            )

            self.last_zone_analysis = zone_analysis

            # Detectar concentración táctica
            for team_id in [0, 1]:
                if team_id not in zone_analysis:
                    continue

                analysis = zone_analysis[team_id]
                concentration = analysis.get('concentration', 0)
                dominant_zones = analysis.get('dominant_zones', [])

                # Alerta si concentración es notable (>60%)
                if concentration >= 0.60 and len(dominant_zones) <= 3:
                    # Limitar frecuencia
                    alert_key = f"{alert_type}_team{team_id}"
                    if not self.can_send_alert(alert_key):
                        continue

                    zones_str = ', '.join(dominant_zones) if dominant_zones else "zona central"

                    alert = self._create_alert(
                        frame_id=frame_id,
                        alert_type="zone",
                        severity="info",
                        title=f"📍 Concentración táctica - Equipo {team_id}",
                        message=f"Equipo {team_id} concentrando su juego en: {zones_str} ({concentration*100:.0f}% de la posesión)",
                        team_id=team_id,
                        data={
                            'dominant_zones': dominant_zones,
                            'concentration': concentration,
                            'pattern': analysis.get('pattern', 'unknown')
                        }
                    )
                    alerts.append(alert)
                    self._mark_alert_sent(alert_key)

        except Exception as e:
            logger.error(f"Error en _check_zone_dominance_patterns: {e}")

        return alerts

    def _check_passing_chain_efficiency(self, frame_id: int, events: Optional[List[Dict]] = None) -> List[Alert]:
        """
        Detecta cadenas de pases efectivas y las comenta.
        """
        alerts = []

        if not self.tactical_analyzer or not events:
            return alerts

        alert_type = "passing_chain"

        try:
            # Procesar eventos de pases
            for event in events:
                if event.get('type') == 'pass':
                    self.tactical_analyzer.insight_generator.chain_detector.update(event, self.fps)

            # Revisar cadenas notables
            for team_id in [0, 1]:
                notable_chains = self.tactical_analyzer.insight_generator.chain_detector.get_notable_chains(
                    team_id, min_length=5
                )

                if notable_chains and self.can_send_alert(f"{alert_type}_team{team_id}"):
                    best_chain = notable_chains[0]

                    zones_str = ', '.join(best_chain.zones) if best_chain.zones else "zona desconocida"

                    alert = self._create_alert(
                        frame_id=frame_id,
                        alert_type="passing",
                        severity="info",
                        title=f"⚡ Cadena de pases efectiva - Equipo {team_id}",
                        message=f"Equipo {team_id} completó una secuencia de {best_chain.length} pases en {zones_str}",
                        team_id=team_id,
                        data={
                            'chain_info': best_chain.to_dict(),
                            'is_active': best_chain.is_active
                        }
                    )
                    alerts.append(alert)
                    self._mark_alert_sent(f"{alert_type}_team{team_id}")

        except Exception as e:
            logger.error(f"Error en _check_passing_chain_efficiency: {e}")

        return alerts

    def _check_tactical_shift_detection(self, frame_id: int) -> List[Alert]:
        """
        Detecta cambios tácticos (cambios en el patrón de zonas).
        """
        alerts = []

        if not self.tactical_analyzer or not self.last_zone_analysis:
            return alerts

        alert_type = "tactical_shift"

        try:
            for team_id in [0, 1]:
                shift = self.tactical_analyzer.insight_generator.zone_analyzer.detect_zone_shift(team_id)

                if shift and self.can_send_alert(f"{alert_type}_team{team_id}"):
                    alert = self._create_alert(
                        frame_id=frame_id,
                        alert_type="tactical",
                        severity="info",
                        title=f"🔄 Cambio táctico - Equipo {team_id}",
                        message=f"Equipo {team_id} está ajustando su táctica: {shift}",
                        team_id=team_id,
                        data={'tactical_shift': shift}
                    )
                    alerts.append(alert)
                    self._mark_alert_sent(f"{alert_type}_team{team_id}")

        except Exception as e:
            logger.error(f"Error en _check_tactical_shift_detection: {e}")

        return alerts

    def _generate_professional_alert(self, frame_id: int, possession_stats: Dict,
                                    spatial_stats: Optional[Dict] = None) -> Optional[Alert]:
        """
        Genera alerta profesional con análisis usando Claude API.

        Se ejecuta solo en momentos clave para economizar API calls.
        """
        if not self.tactical_analyzer or not self.tactical_analyzer.narrative_generator.should_generate():
            return None

        try:
            # Obtener contexto
            possession = possession_stats.get('possession_percent', {}) or {}
            zones = self.last_zone_analysis or {}
            chains = {}

            if spatial_stats:
                for team_id in [0, 1]:
                    active_chain = self.tactical_analyzer.insight_generator.chain_detector.get_active_chain(team_id)
                    chains[team_id] = {
                        'active': active_chain.to_dict() if active_chain else None,
                        'stats': self.tactical_analyzer.insight_generator.chain_detector.get_chain_stats(team_id)
                    }

            # Determinar tipo de evento
            event_type = self._determine_significant_event(possession_stats)

            context = {
                'possession': possession,
                'zones': zones,
                'chains': chains,
                'event_type': event_type,
                'time_minutes': int((frame_id / self.fps) / 60)
            }

            # Llamar a Claude API (será async pero podemos hacer await aquí)
            # Para compatibilidad con código síncrono, usamos asyncio.run
            try:
                narrative = asyncio.run(
                    self.tactical_analyzer.narrative_generator.generate_narrative(context)
                )
            except RuntimeError:
                # Si ya hay un loop running, usar otra estrategia
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # EnQueue como tarea pero no esperamos
                    logger.warning("Event loop already running, skipping narrative for this frame")
                    return None
                else:
                    narrative = asyncio.run(
                        self.tactical_analyzer.narrative_generator.generate_narrative(context)
                    )

            if narrative:
                alert = self._create_alert(
                    frame_id=frame_id,
                    alert_type="tactical",
                    severity="info",
                    title="📊 Análisis profesional",
                    message=narrative,
                    data={
                        'narrative': narrative,
                        'event_type': event_type,
                        'model': 'claude-api'
                    }
                )
                return alert

        except Exception as e:
            logger.error(f"Error generating professional alert: {e}")

        return None

    def _determine_significant_event(self, possession_stats: Dict) -> str:
        """Determina el tipo de evento significativo"""
        # Obtener cambios recientes
        if len(self.possession_history) < 2:
            return "momentum_check"

        prev_poss = self.possession_history[-2][1]
        curr_poss = possession_stats.get('possession_percent', {})

        # Detectar cambios grandes en posesión
        for team_id in [0, 1]:
            swing = abs(curr_poss.get(team_id, 0) - prev_poss.get(team_id, 0))
            if swing >= 15:
                return f"possession_shift_team_{team_id}"

        return "regular_check"

    def get_alert_summary(self) -> Dict:
        """Retorna resumen de alertas generadas"""
        return {
            'total_alerts': self.alert_counter,
            'checks_performed': len(self.possession_history),
            'last_check_frame': self.last_check_frame
        }
