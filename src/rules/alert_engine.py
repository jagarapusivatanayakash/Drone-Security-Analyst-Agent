"""
SIMPLE Alert Rules for Theft & Accident Detection
"""

from typing import List, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlertRulesEngine:
    """Simple rule-based alerts for theft and accidents"""

    def __init__(self):
        logger.info("Alert Rules Engine initialized (Simple Mode)")

    def evaluate_frame(
        self,
        frame_data: Dict[str, Any],
        description: str,
        past_frames: List[Dict[str, Any]],
        analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Check for theft or accident in frame"""

        alerts = []
        desc_lower = description.lower()

        # ============ THEFT DETECTION ============

        # 1. GUN/WEAPON - CRITICAL
        weapon_words = ["gun", "weapon", "knife", "armed", "firearm", "pistol", "rifle"]
        if any(word in desc_lower for word in weapon_words):
            alerts.append(
                {
                    "rule": "weapon_detected",
                    "severity": "CRITICAL",
                    "message": f"⚠️ WEAPON DETECTED: {description[:100]}",
                    "timestamp": frame_data["timestamp"],
                    "frame_id": frame_data["frame_id"],
                }
            )

        # 2. LOITERING - Check if person appears in multiple frames
        if "person" in desc_lower or "people" in desc_lower:
            person_count = sum(
                1
                for f in past_frames[-5:]
                if "person" in f.get("description", "").lower()
            )

            if person_count >= 3:  # Person in 3+ consecutive frames
                alerts.append(
                    {
                        "rule": "loitering",
                        "severity": "MEDIUM",
                        "message": f"Person loitering at {frame_data['telemetry'].get('location', 'location')}",
                        "timestamp": frame_data["timestamp"],
                        "frame_id": frame_data["frame_id"],
                    }
                )

        # 3. SUSPICIOUS BEHAVIOR - Only flag truly suspicious actions
        suspicious_words = [
            "breaking",
            "forcing",
            "smashing",
            "prying",
            "tampering",
            "vandalizing",
            "climbing fence",
            "jumping over",
        ]
        if any(word in desc_lower for word in suspicious_words):
            alerts.append(
                {
                    "rule": "suspicious_activity",
                    "severity": "HIGH",
                    "message": f"Suspicious activity: {description[:100]}",
                    "timestamp": frame_data["timestamp"],
                    "frame_id": frame_data["frame_id"],
                }
            )

        # 4. RUNNING (possible theft escape)
        running_words = ["running", "fleeing", "rushing", "sprinting"]
        if any(word in desc_lower for word in running_words):
            alerts.append(
                {
                    "rule": "running_person",
                    "severity": "HIGH",
                    "message": f"Person running detected - possible theft",
                    "timestamp": frame_data["timestamp"],
                    "frame_id": frame_data["frame_id"],
                }
            )

        # ============ ACCIDENT DETECTION ============

        # 1. CAR COLLISION - CRITICAL
        collision_words = [
            "collision",
            "crash",
            "colliding",
            "crashed",
            "hit",
            "accident",
            "impact",
            "smash",
        ]
        if any(word in desc_lower for word in collision_words):
            alerts.append(
                {
                    "rule": "collision_detected",
                    "severity": "CRITICAL",
                    "message": f"🚨 ACCIDENT: {description[:100]}",
                    "timestamp": frame_data["timestamp"],
                    "frame_id": frame_data["frame_id"],
                }
            )

        # 2. PERSON INJURED
        injury_words = ["injured", "hurt", "fallen", "lying", "down", "unconscious"]
        if any(word in desc_lower for word in injury_words):
            alerts.append(
                {
                    "rule": "injury_detected",
                    "severity": "CRITICAL",
                    "message": f"🚑 INJURY: Person may be injured - {description[:100]}",
                    "timestamp": frame_data["timestamp"],
                    "frame_id": frame_data["frame_id"],
                }
            )

        # 3. DAMAGED VEHICLE
        damage_words = ["damaged", "broken", "dented", "debris", "glass"]
        vehicle_words = ["car", "vehicle", "truck"]
        if any(d in desc_lower for d in damage_words) and any(
            v in desc_lower for v in vehicle_words
        ):
            alerts.append(
                {
                    "rule": "vehicle_damage",
                    "severity": "MEDIUM",
                    "message": f"Vehicle damage detected: {description[:80]}",
                    "timestamp": frame_data["timestamp"],
                    "frame_id": frame_data["frame_id"],
                }
            )

        # 4. CROWD GATHERING (after accident)
        crowd_words = ["crowd", "people gathered", "group", "multiple people"]
        if any(word in desc_lower for word in crowd_words):
            # Check if accident happened recently
            recent_accident = any(
                "collision" in f.get("description", "").lower()
                for f in past_frames[-10:]
            )
            if recent_accident:
                alerts.append(
                    {
                        "rule": "crowd_at_accident",
                        "severity": "MEDIUM",
                        "message": "Crowd gathering at accident scene",
                        "timestamp": frame_data["timestamp"],
                        "frame_id": frame_data["frame_id"],
                    }
                )

        # Log alerts
        for alert in alerts:
            logger.warning(
                f"🚨 [{alert['severity']}] {alert['message']} at {alert['timestamp']}"
            )

        return alerts
