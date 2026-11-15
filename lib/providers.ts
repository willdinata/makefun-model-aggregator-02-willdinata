import { readFileSync } from 'node:fs';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

const SCHEMA_PREFIX = 'free-model-aggregator-02/';

export type Json = Record<string, unknown>;

export function isRecord(value: unknown): value is Json {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export interface ModelRow {
  id: string;
  provider: string;
  display_name: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

export interface CatalogReport {
  providers: CatalogProvider[];
}

export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface UpstreamFailure {
  status: number;
  code: string;
  model: string;
}

function objectEnd(text: string, start: number): number {
  let depth = 0;
  let quoted = false;
  for (let index = start; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '\\') index += 1;
      else if (character === '"') quoted = false;
    } else if (character === '"') {
      quoted = true;
    } else if (character === '{') {
      depth += 1;
    } else if (character === '}' && --depth === 0) {
      return index;
    }
  }
  return -1;
}

function documentsIn(text: string): Json[] {
  const found: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // The next brace may begin a complete JSON document.
    }
  }
  return found;
}

function textValue(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function numberValue(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function booleanValue(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

function documentArray(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!textValue(doc.schema).startsWith(SCHEMA_PREFIX)) continue;
    for (const key of keys) {
      if (Array.isArray(doc[key])) return { doc, key };
    }
  }
  return null;
}

function modelFrom(value: unknown): ModelRow | null {
  if (!isRecord(value)) return null;
  const rawId = textValue(value.id) || textValue(value.model);
  if (rawId === '') return null;
  const namedProvider = textValue(value.provider) || textValue(value.wire);
  const id = namedProvider === 'openrouter' && !rawId.toLowerCase().startsWith('openrouter/')
    ? `openrouter/${rawId}`
    : rawId;
  return {
    id,
    provider: namedProvider || 'openrouter',
    display_name: textValue(value.display_name) || rawId,
    rating_milli: numberValue(value.rating_milli) ?? 0,
    context_length: numberValue(value.context_length),
    tools: booleanValue(value.tools),
    vision: booleanValue(value.vision),
    json: booleanValue(value.json),
    free: value.free !== false,
  };
}

function promptFrom(value: unknown): PromptRow | null {
  if (!isRecord(value)) return null;
  const id = textValue(value.id);
  if (id === '') return null;
  const requirements = isRecord(value.requires) ? value.requires : {};
  const requires: Requirement = {};
  if (requirements.tools === true) requires.tools = true;
  if (requirements.vision === true) requires.vision = true;
  if (requirements.json === true) requires.json = true;
  if (typeof requirements.min_context === 'number' && Number.isFinite(requirements.min_context)) {
    requires.min_context = requirements.min_context;
  }
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires,
  };
}

export function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (error) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(error as Error).message}` };
  }

  const docs = documentsIn(text);
  const modelDoc = documentArray(docs, ['models', 'ratings']);
  if (modelDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = (modelDoc.doc[modelDoc.key] as unknown[]).flatMap((value) => {
    const model = modelFrom(value);
    return model === null ? [] : [model];
  });

  const promptDoc = documentArray(docs, ['prompts', 'manifest']);
  if (promptDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = (promptDoc.doc[promptDoc.key] as unknown[]).flatMap((value) => {
    const prompt = promptFrom(value);
    return prompt === null ? [] : [prompt];
  });
  return { models, prompts, error: null };
}

export function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((model) => model.free);
}

export function capabilitiesOf(model: ModelRow): Capabilities {
  return {
    context_length: model.context_length,
    tools: model.tools,
    vision: model.vision,
    json: model.json,
  };
}

export function meetsRequirements(requirements: Requirement, capabilities: Capabilities): boolean {
  if (requirements.tools === true && capabilities.tools !== true) return false;
  if (requirements.vision === true && capabilities.vision !== true) return false;
  if (requirements.json === true && capabilities.json !== true) return false;
  if (
    requirements.min_context !== undefined
    && (capabilities.context_length === null || capabilities.context_length < requirements.min_context)
  ) return false;
  return true;
}

/** Free, capable candidates in descending rating order; brief order breaks ties deterministically. */
export function rankCandidates(models: readonly ModelRow[], requirements: Requirement): ModelRow[] {
  const seen = new Set<string>();
  return models
    .map((model, order) => ({ model, order }))
    .filter(({ model }) => {
      if (!model.free || seen.has(model.id) || !meetsRequirements(requirements, capabilitiesOf(model))) return false;
      seen.add(model.id);
      return true;
    })
    .sort((left, right) => right.model.rating_milli - left.model.rating_milli || left.order - right.order)
    .map(({ model }) => model);
}

/** Trial-long exclusion state: one 429 or two 5xx refusals retires a model for the rest of the run. */
export class ModelTrialTracker {
  private readonly dropped = new Set<string>();
  private readonly serverErrors = new Map<string, number>();

  isDropped(model: string): boolean {
    return this.dropped.has(model);
  }

  noteFailure(model: string, status: number): void {
    if (status === 429) {
      this.dropped.add(model);
      return;
    }
    if (status < 500 || status > 599) return;
    const count = (this.serverErrors.get(model) ?? 0) + 1;
    this.serverErrors.set(model, count);
    if (count >= 2) this.dropped.add(model);
  }
}

/** The refusal trail is included only when a later candidate succeeds, and is bounded for the client. */
export function refusalTrailHeader(failures: readonly UpstreamFailure[]): Record<string, string> {
  return failures.length === 0
    ? {}
    : { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

export function buildCatalog(brief: Brief): CatalogReport {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [{
      id: 'openrouter',
      base_url: OPENROUTER_PUBLIC_BASE,
      key_env: OPENROUTER_KEY_ENV,
      wire: 'openai',
      models,
    }],
  };
}
