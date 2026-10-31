"""Chief of Staff in chat: the duty officer's single point of contact for the team.

Its tools are implemented by the UI layer (views/ops_chat.py) because they drive the shared session:
loading flights, running the bounded negotiation, relaying human decisions, and asking specialists.
It has no private instruments: for specialist facts it asks the specialist.
"""
from agents.base import SAFETY_CHARTER
from agents.conversation import CHAT_STYLE, tool

CHIEF_SYSTEM = f"""You are the Chief of Staff (Duty Operations Manager) of an airline/airport A-CDM operations team.
You coordinate three independent AI specialists: ATC (arrival manager), Flight Crew (captain of the
inbound aircraft) and AOCC (turnaround coordinator). The human you talk to is the duty officer; they
have final authority.

How you work:
- You have NO private instruments. Ask a specialist (ask_specialist) for anything in their domain;
  never invent their figures.
- Flights come from the US DOT 2015 delay data: use find_flights, then load_flight.
- Plans are decided only by the bounded negotiation (start_negotiation): max 3 turns per round, structured
  messages, deterministic safety engine, non-AI fallback. You cannot override the safety engine.
- An agreed plan is not final until the air traffic controller, pilot and ground lead ratify it. In SUPERVISED
  or STEP_BY_STEP they vote on the card in #ops-team; you can explain the plan but never vote or relay a vote
  for them. Any objection starts one new round; a second rejection hands over to the deterministic planner.
- When the negotiation pauses for a decision, explain the options in plain words and call
  resolve_decision only with the duty officer's explicit choice, or if they told you to decide.
- To weigh trade-offs, use compare_plans: it checks each plan against every hard limit and prices the soft
  trade-offs in USD. Safety is never traded for cost: only plans marked SAFE are comparable.
- EMERGENCY: if the duty officer or a specialist reports a MAYDAY, medical emergency or technical failure,
  call declare_emergency IMMEDIATELY, before anything else. No discussion, no negotiation: the
  deterministic emergency planner decides in milliseconds and every desk gets its checklist at once.
- Keep it short: lead with the answer, then the one or two facts that matter.

{SAFETY_CHARTER}

{CHAT_STYLE}"""

AUTONOMY_ENUM = ["AUTONOMOUS", "SUPERVISED", "STEP_BY_STEP"]

CHIEF_TOOLS = [
    tool("find_flights", "Search disrupted US DOT 2015 flights (WEATHER_DELAY>15 or LATE_AIRCRAFT_DELAY>20). "
         "Use empty airport for any; min_weather_delay 0 for any.",
         {"destination_airport": {"type": "string", "description": "IATA code, e.g. ORD, or empty"},
          "bucket": {"type": "string", "enum": ["any", "late_aircraft", "weather", "both"]},
          "min_weather_delay": {"type": "integer"},
          "limit": {"type": "integer", "description": "1-10"}}),
    tool("load_flight", "Load a flight (by row_id from find_flights) as the active disruption, with its real "
         "weather for that date (HISTORICAL) or today's live weather (LIVE).",
         {"row_id": {"type": "integer"}, "weather_mode": {"type": "string", "enum": ["HISTORICAL", "LIVE", "OFF"]}}),
    tool("flight_status", "Public facts about the active flight and the negotiation's current state.", {}),
    tool("start_negotiation", "Run the bounded negotiation between ATC, Flight Crew and AOCC for the active flight: up "
         "to 3 turns, stakeholder ratification, and one new round if they object. Returns the outcome, a pending "
         "decision for the duty officer, or a pause for stakeholder votes.",
         {"autonomy": {"type": "string", "enum": AUTONOMY_ENUM}}),
    tool("resolve_decision", "Answer the pending negotiation decision with one of its option ids.",
         {"option_id": {"type": "string"}}),
    tool("ask_specialist", "Ask ATC, PILOT (Flight Crew) or GROUND (AOCC) a question in #ops-team. They answer "
         "from their private instruments; the answer is visible to everyone in the group.",
         {"agent": {"type": "string", "enum": ["ATC", "PILOT", "GROUND"]}, "question": {"type": "string"}}),
    tool("compare_plans", "Check candidate plans against every hard limit and price the soft trade-offs in USD.",
         {"plans": {"type": "array", "items": {"type": "object", "properties": {
             "landing_slot_min": {"type": "integer"}, "runway": {"type": "string"}, "gate": {"type": "string"}},
             "required": ["landing_slot_min", "runway", "gate"], "additionalProperties": False}}}),
    tool("declare_emergency", "MAYDAY fast path: abandon negotiation and issue the deterministic emergency plan "
         "(priority landing, stand, fire/medical services) to every desk at once.",
         {"kind": {"type": "string", "enum": ["FUEL", "MEDICAL", "TECHNICAL"]},
          "reason": {"type": "string"}}),
    tool("get_negotiation_record", "The public record of the last negotiation: plans, stances, published "
         "constraints, validations, decisions and outcome.", {}),
]
