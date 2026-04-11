"""
Prompt template service for configurable LLM prompts.

Loads prompts from prompt_templates table with fallback hierarchy:
    agent → tenant → global default

Supports Jinja2 templating for dynamic prompt generation.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import hashlib

from jinja2 import Environment, BaseLoader, TemplateSyntaxError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("gami.prompt_service")


# Default prompts (fallback when no database entry exists)
DEFAULT_PROMPTS: Dict[str, Dict[str, Any]] = {
    "entity_extraction": {
        "system_prompt": """You are a precise entity extraction system. Extract named entities from text.

Rules:
- Extract only concrete, identifiable entities (not abstract concepts)
- Normalize entity names (consistent capitalization, no typos)
- Assign appropriate types: person, organization, service, database, file, host, config, api, event, concept
- Return ONLY valid JSON arrays with no additional text""",
        "user_prompt_template": """Extract entities from the following text:

{{ text }}

Return a JSON array of objects with keys: name, type, description
Example: [{"name": "PostgreSQL", "type": "database", "description": "Primary data store"}]""",
        "temperature": 0.1,
        "max_tokens": 2048,
    },
    "claim_extraction": {
        "system_prompt": """You are a precise claim extraction system. Extract factual assertions from text.

Rules:
- Extract specific, verifiable claims (not opinions or generalizations)
- Each claim should be a complete assertion that can be true or false
- Include subject, predicate, and object for each claim
- Assign confidence based on how explicitly the claim is stated
- Return ONLY valid JSON arrays with no additional text""",
        "user_prompt_template": """Extract factual claims from:

{{ text }}

Return a JSON array with keys: subject, predicate, object, confidence (0.0-1.0), modality (fact|possibility|requirement)
Example: [{"subject": "Redis", "predicate": "runs_on_port", "object": "6379", "confidence": 0.95, "modality": "fact"}]""",
        "temperature": 0.1,
        "max_tokens": 2048,
    },
    "relation_extraction": {
        "system_prompt": """You are a relation extraction system. Identify relationships between entities.

Rules:
- Only extract relationships explicitly stated or strongly implied
- Use standardized relation types when possible
- Include confidence scores based on explicitness
- Return ONLY valid JSON arrays""",
        "user_prompt_template": """Given these entities:
{% for entity in entities %}
- {{ entity.name }} ({{ entity.type }})
{% endfor %}

Extract relationships from:
{{ text }}

Return JSON array: [{"source": "entity1", "target": "entity2", "relation_type": "uses", "confidence": 0.9}]""",
        "temperature": 0.1,
        "max_tokens": 2048,
    },
    "causal_extraction": {
        "system_prompt": """You are a causal relationship extraction system. Identify cause-effect relationships.

Rules:
- Look for explicit causal language: because, caused by, resulted in, led to, due to, therefore
- Also identify implicit causality from temporal sequences with clear effects
- Validate temporal ordering when timestamps are available
- Include confidence based on how explicitly the causality is stated
- Return ONLY valid JSON arrays""",
        "user_prompt_template": """Extract causal relationships from:

{{ text }}

{% if entities %}
Known entities: {{ entities | join(', ') }}
{% endif %}

Return JSON array: [{"cause": "text", "effect": "text", "causal_type": "because|caused_by|resulted_in|led_to|triggered", "confidence": 0.8, "cause_entity": "optional", "effect_entity": "optional"}]""",
        "temperature": 0.2,
        "max_tokens": 2048,
    },
    "event_extraction": {
        "system_prompt": """You are an event extraction system. Identify discrete events with temporal context.

Rules:
- Extract events with clear boundaries (start, end, or point in time)
- Include participants, actions, and outcomes
- Normalize timestamps to ISO format when possible
- Return ONLY valid JSON arrays""",
        "user_prompt_template": """Extract events from:

{{ text }}

Return JSON array: [{"event_type": "deployment|incident|configuration_change|meeting|decision", "description": "...", "timestamp": "ISO or relative", "participants": ["..."], "outcome": "..."}]""",
        "temperature": 0.1,
        "max_tokens": 2048,
    },
    "summarization": {
        "system_prompt": """You are a summarization system. Create concise, accurate summaries.

Rules:
- Preserve key information and relationships
- Use clear, direct language
- Maintain factual accuracy
- Include relevant context""",
        "user_prompt_template": """Summarize the following {% if summary_type %}{{ summary_type }}{% endif %}:

{{ text }}

{% if max_length %}Maximum length: {{ max_length }} words{% endif %}
{% if focus_entities %}Focus on: {{ focus_entities | join(', ') }}{% endif %}""",
        "temperature": 0.3,
        "max_tokens": 1024,
    },
    "state_classification": {
        "system_prompt": """You are a conversation state classifier. Analyze message intent and context.

States:
- idle: No active task or direction
- debugging: Investigating errors, bugs, or unexpected behavior
- planning: Discussing approach, design, or strategy
- recalling: Retrieving past information or context
- exploring: Learning about concepts or systems
- executing: Actively performing requested actions
- confirming: Verifying understanding or results

Return ONLY the state name and confidence score.""",
        "user_prompt_template": """Classify the conversation state from these recent messages:

{% for msg in messages %}
[{{ msg.role }}]: {{ msg.content[:200] }}
{% endfor %}

Return JSON: {"state": "debugging|planning|recalling|exploring|executing|confirming|idle", "confidence": 0.0-1.0}""",
        "temperature": 0.2,
        "max_tokens": 100,
    },
    "abstraction_generation": {
        "system_prompt": """You are a memory abstraction system. Create generalized summaries from multiple related memories.

Rules:
- Identify common patterns and themes
- Preserve essential information while removing redundancy
- Create a single coherent abstraction
- Maintain factual accuracy""",
        "user_prompt_template": """Create an abstraction from these related memories:

{% for text in texts %}
Memory {{ loop.index }}: {{ text }}
---
{% endfor %}

Return a single paragraph that captures the essential shared information.""",
        "temperature": 0.4,
        "max_tokens": 500,
    },
    "contradiction_detection": {
        "system_prompt": """You are a contradiction detection system. Identify conflicting claims.

Rules:
- Compare claims for logical inconsistency
- Consider temporal context (newer may supersede older)
- Identify partial vs complete contradictions
- Return specific explanation of the contradiction""",
        "user_prompt_template": """Check if these claims contradict:

Claim A: {{ claim_a }}
Claim B: {{ claim_b }}

Return JSON: {"contradicts": true|false, "explanation": "...", "severity": "full|partial|supersedes"}""",
        "temperature": 0.1,
        "max_tokens": 300,
    },
    "inference_generation": {
        "system_prompt": """You are an inference generation system. Derive new knowledge from existing facts.

Rules:
- Only generate logically valid inferences
- Mark confidence based on inference chain strength
- Flag speculative inferences clearly
- Do not generate facts not derivable from premises""",
        "user_prompt_template": """Given these established facts:
{% for fact in facts %}
- {{ fact }}
{% endfor %}

Generate valid inferences. Return JSON array:
[{"inference": "...", "confidence": 0.0-1.0, "premises_used": ["..."], "inference_type": "deductive|inductive|abductive"}]""",
        "temperature": 0.3,
        "max_tokens": 1024,
    },
}


@dataclass
class PromptTemplate:
    """Loaded prompt template with metadata."""
    template_id: str
    template_type: str
    system_prompt: str
    user_prompt_template: str
    temperature: float
    max_tokens: int
    model_override: Optional[str]
    output_format: str
    output_schema: Optional[Dict]
    source: str  # "database" or "default"


class PromptTemplateError(Exception):
    """Error in prompt template processing."""
    pass


class PromptService:
    """
    Service for loading and rendering prompt templates.

    Supports hierarchical loading with caching:
    - Agent-specific prompts override tenant prompts
    - Tenant prompts override global prompts
    - Global prompts override hardcoded defaults
    """

    def __init__(
        self,
        cache_ttl_seconds: int = 300,
        validate_on_load: bool = True,
    ):
        """Initialize the prompt service.

        Args:
            cache_ttl_seconds: How long to cache loaded prompts
            validate_on_load: Whether to validate Jinja2 syntax on load
        """
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.validate_on_load = validate_on_load

        # Cache: {cache_key: (template, loaded_at)}
        self._cache: Dict[str, Tuple[PromptTemplate, datetime]] = {}

        # Jinja2 environment
        self._jinja_env = Environment(
            loader=BaseLoader(),
            autoescape=False,  # No HTML escaping for prompts
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add custom filters
        self._jinja_env.filters['truncate'] = self._truncate_filter
        self._jinja_env.filters['json_safe'] = self._json_safe_filter

    def _truncate_filter(self, text: str, length: int = 200) -> str:
        """Truncate text to specified length."""
        if len(text) <= length:
            return text
        return text[:length-3] + "..."

    def _json_safe_filter(self, text: str) -> str:
        """Escape text for JSON embedding."""
        return text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

    def _cache_key(
        self,
        template_type: str,
        tenant_id: str,
        agent_id: Optional[str],
    ) -> str:
        """Generate cache key."""
        return f"{template_type}:{tenant_id}:{agent_id or '*'}"

    async def get_template(
        self,
        db: AsyncSession,
        template_type: str,
        tenant_id: str = "*",
        agent_id: Optional[str] = None,
    ) -> PromptTemplate:
        """Load a prompt template with hierarchical lookup.

        Args:
            db: Database session
            template_type: Type of template (entity_extraction, claim_extraction, etc.)
            tenant_id: Tenant ID for tenant-specific prompts
            agent_id: Agent ID for agent-specific prompts

        Returns:
            PromptTemplate with resolved prompt content

        Raises:
            PromptTemplateError: If template not found and no default exists
        """
        cache_key = self._cache_key(template_type, tenant_id, agent_id)

        # Check cache
        if cache_key in self._cache:
            template, loaded_at = self._cache[cache_key]
            if datetime.now() - loaded_at < self.cache_ttl:
                return template

        # Try database lookup with fallback hierarchy
        template = await self._load_from_db(db, template_type, tenant_id, agent_id)

        if template is None:
            # Fall back to default
            template = self._get_default_template(template_type)

        if template is None:
            raise PromptTemplateError(f"No template found for type: {template_type}")

        # Validate Jinja2 syntax
        if self.validate_on_load:
            self._validate_template(template)

        # Cache it
        self._cache[cache_key] = (template, datetime.now())

        return template

    async def _load_from_db(
        self,
        db: AsyncSession,
        template_type: str,
        tenant_id: str,
        agent_id: Optional[str],
    ) -> Optional[PromptTemplate]:
        """Load template from database with fallback hierarchy."""

        # Build query to find most specific matching template
        query = text("""
            SELECT
                template_id,
                template_type,
                system_prompt,
                user_prompt_template,
                temperature,
                max_tokens,
                model_override,
                output_format,
                output_schema,
                tenant_id,
                agent_id
            FROM prompt_templates
            WHERE template_type = :template_type
            AND is_active = TRUE
            AND (
                (tenant_id = :tenant_id AND agent_id = :agent_id)
                OR (tenant_id = :tenant_id AND agent_id IS NULL)
                OR (tenant_id = '*' AND agent_id IS NULL)
            )
            ORDER BY
                CASE
                    WHEN tenant_id = :tenant_id AND agent_id = :agent_id THEN 1
                    WHEN tenant_id = :tenant_id AND agent_id IS NULL THEN 2
                    WHEN tenant_id = '*' AND agent_id IS NULL THEN 3
                END
            LIMIT 1
        """)

        result = await db.execute(query, {
            "template_type": template_type,
            "tenant_id": tenant_id,
            "agent_id": agent_id,
        })
        row = result.fetchone()

        if row is None:
            return None

        return PromptTemplate(
            template_id=row.template_id,
            template_type=row.template_type,
            system_prompt=row.system_prompt or "",
            user_prompt_template=row.user_prompt_template or "",
            temperature=row.temperature or 0.1,
            max_tokens=row.max_tokens or 2048,
            model_override=row.model_override,
            output_format=row.output_format or "json",
            output_schema=row.output_schema,
            source="database",
        )

    def _get_default_template(self, template_type: str) -> Optional[PromptTemplate]:
        """Get hardcoded default template."""
        if template_type not in DEFAULT_PROMPTS:
            return None

        default = DEFAULT_PROMPTS[template_type]

        return PromptTemplate(
            template_id=f"default_{template_type}",
            template_type=template_type,
            system_prompt=default["system_prompt"],
            user_prompt_template=default["user_prompt_template"],
            temperature=default.get("temperature", 0.1),
            max_tokens=default.get("max_tokens", 2048),
            model_override=None,
            output_format="json",
            output_schema=None,
            source="default",
        )

    def _validate_template(self, template: PromptTemplate) -> None:
        """Validate Jinja2 syntax in template."""
        try:
            self._jinja_env.from_string(template.user_prompt_template)
        except TemplateSyntaxError as e:
            raise PromptTemplateError(
                f"Invalid Jinja2 in template {template.template_id}: {e}"
            )

    def render(
        self,
        template: PromptTemplate,
        variables: Dict[str, Any],
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Render a prompt template with variables.

        Args:
            template: The loaded template
            variables: Variables for Jinja2 rendering

        Returns:
            Tuple of (system_prompt, rendered_user_prompt, llm_params)
        """
        try:
            jinja_template = self._jinja_env.from_string(template.user_prompt_template)
            rendered_user = jinja_template.render(**variables)
        except Exception as e:
            raise PromptTemplateError(f"Failed to render template: {e}")

        llm_params = {
            "temperature": template.temperature,
            "max_tokens": template.max_tokens,
        }
        if template.model_override:
            llm_params["model"] = template.model_override

        return template.system_prompt, rendered_user, llm_params

    async def get_prompt(
        self,
        db: AsyncSession,
        template_type: str,
        variables: Dict[str, Any],
        tenant_id: str = "*",
        agent_id: Optional[str] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Convenience method to load and render a prompt.

        Args:
            db: Database session
            template_type: Type of template
            variables: Variables for rendering
            tenant_id: Tenant for lookup
            agent_id: Agent for lookup

        Returns:
            Tuple of (system_prompt, rendered_user_prompt, llm_params)
        """
        template = await self.get_template(db, template_type, tenant_id, agent_id)
        return self.render(template, variables)

    def invalidate_cache(
        self,
        template_type: Optional[str] = None,
        tenant_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> int:
        """Invalidate cached templates.

        Args:
            template_type: If provided, only invalidate this type
            tenant_id: If provided, only invalidate for this tenant
            agent_id: If provided, only invalidate for this agent

        Returns:
            Number of cache entries invalidated
        """
        if template_type is None and tenant_id is None and agent_id is None:
            count = len(self._cache)
            self._cache.clear()
            return count

        keys_to_remove = []
        for key in self._cache:
            parts = key.split(":")
            if len(parts) != 3:
                continue

            t_type, t_tenant, t_agent = parts

            if template_type and t_type != template_type:
                continue
            if tenant_id and t_tenant != tenant_id:
                continue
            if agent_id and t_agent != agent_id and t_agent != "*":
                continue

            keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._cache[key]

        return len(keys_to_remove)


# =============================================================================
# SYNCHRONOUS HELPERS FOR NON-ASYNC CONTEXTS
# =============================================================================

def get_default_prompt(
    template_type: str,
    variables: Dict[str, Any],
) -> Tuple[str, str, Dict[str, Any]]:
    """Get and render a default prompt (no database lookup).

    Useful for synchronous contexts where database access isn't available.

    Args:
        template_type: Type of template
        variables: Variables for rendering

    Returns:
        Tuple of (system_prompt, rendered_user_prompt, llm_params)
    """
    service = PromptService(validate_on_load=False)
    template = service._get_default_template(template_type)

    if template is None:
        raise PromptTemplateError(f"No default template for type: {template_type}")

    return service.render(template, variables)


# =============================================================================
# ADMIN FUNCTIONS
# =============================================================================

async def seed_default_prompts(db: AsyncSession) -> int:
    """Seed default prompts to database.

    Only inserts if no global default exists for each type.

    Returns:
        Number of prompts seeded
    """
    seeded = 0

    for template_type, config in DEFAULT_PROMPTS.items():
        # Check if global default exists
        check = await db.execute(text("""
            SELECT 1 FROM prompt_templates
            WHERE template_type = :type AND tenant_id = '*' AND agent_id IS NULL
        """), {"type": template_type})

        if check.fetchone() is not None:
            continue

        # Insert default
        template_id = f"default_{template_type}"
        await db.execute(text("""
            INSERT INTO prompt_templates (
                template_id, tenant_id, template_name, template_type,
                system_prompt, user_prompt_template,
                temperature, max_tokens, output_format
            ) VALUES (
                :id, '*', :name, :type,
                :system, :user,
                :temp, :tokens, 'json'
            )
        """), {
            "id": template_id,
            "name": f"Default {template_type.replace('_', ' ').title()}",
            "type": template_type,
            "system": config["system_prompt"],
            "user": config["user_prompt_template"],
            "temp": config.get("temperature", 0.1),
            "tokens": config.get("max_tokens", 2048),
        })
        seeded += 1

    await db.commit()
    logger.info(f"Seeded {seeded} default prompt templates")
    return seeded


async def list_templates(
    db: AsyncSession,
    tenant_id: Optional[str] = None,
    template_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List all prompt templates.

    Args:
        db: Database session
        tenant_id: Filter by tenant
        template_type: Filter by type

    Returns:
        List of template metadata
    """
    conditions = ["is_active = TRUE"]
    params = {}

    if tenant_id:
        conditions.append("tenant_id = :tenant_id")
        params["tenant_id"] = tenant_id

    if template_type:
        conditions.append("template_type = :template_type")
        params["template_type"] = template_type

    query = text(f"""
        SELECT
            template_id, tenant_id, agent_id, template_name, template_type,
            temperature, max_tokens, model_override, version, created_at
        FROM prompt_templates
        WHERE {' AND '.join(conditions)}
        ORDER BY template_type, tenant_id, agent_id
    """)

    result = await db.execute(query, params)

    return [
        {
            "template_id": row.template_id,
            "tenant_id": row.tenant_id,
            "agent_id": row.agent_id,
            "template_name": row.template_name,
            "template_type": row.template_type,
            "temperature": row.temperature,
            "max_tokens": row.max_tokens,
            "model_override": row.model_override,
            "version": row.version,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in result.fetchall()
    ]
