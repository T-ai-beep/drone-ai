from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import os
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class MissionData:
    scenario: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    location: str = ""
    unaccounted: int = 0
    located: int = 0
    recovered: int = 0
    deceased: int = 0
    structures_assessed: int = 0
    structures_critical: int = 0
    hazard_zones: int = 0
    survivor_candidates: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    failure_events: list = field(default_factory=list)
    models_active: list = field(default_factory=list)
    data_sources: list = field(default_factory=list)
    max_altitude: float = 0.0
    flight_duration_seconds: int = 0
    battery_start: float = 0.0
    battery_end: float = 0.0
    notes: str = ""

def generate_report(mission_data: MissionData, filename="mission_report.pdf"):
    
