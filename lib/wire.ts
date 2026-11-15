import { isRecord, type Json } from './providers.ts';

export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

const FIXED_OPENAI_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((model) => ({
      id: model.id,
      object: 'model',
      created: FIXED_OPENAI_CREATED,
      owned_by: model.provider,
    })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((model) => ({
      type: 'model',
      id: model.id,
      display_name: model.display_name,
      created_at: FIXED_ANTHROPIC_CREATED_AT,
    })),
    has_more: false,
  };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((part) => isRecord(part) && part.type === 'text' && typeof part.text === 'string' ? [part.text] : [])
    .join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function objectOfJson(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value !== 'string' || value.trim() === '') return {};
  try {
    const parsed: unknown = JSON.parse(value);
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function anthropicBlockToOpenAI(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') {
    return { type: 'text', text: block.text };
  }
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const mediaType = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${mediaType};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

function openAIPartToAnthropic(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type !== 'image_url' || !isRecord(part.image_url) || typeof part.image_url.url !== 'string') return null;
  const url = part.image_url.url;
  const match = /^data:([^;,]+);base64,([\s\S]*)$/.exec(url);
  if (match !== null) {
    return { type: 'image', source: { type: 'base64', media_type: match[1], data: match[2] } };
  }
  return { type: 'image', source: { type: 'url', url } };
}

function openAIContentToAnthropic(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ type: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  return content.flatMap((part) => isRecord(part) ? [openAIPartToAnthropic(part)].filter((value): value is Json => value !== null) : []);
}

function assistantOpenAIMessageToAnthropic(message: Json): Json {
  const blocks = openAIContentToAnthropic(message.content);
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const raw of calls) {
    if (!isRecord(raw)) continue;
    const fn = isRecord(raw.function) ? raw.function : {};
    blocks.push({
      type: 'tool_use',
      id: typeof raw.id === 'string' ? raw.id : '',
      name: typeof fn.name === 'string' ? fn.name : '',
      input: objectOfJson(fn.arguments),
    });
  }
  return { role: 'assistant', content: blocks };
}

/** Convert an Anthropic Messages request to the OpenAI Chat Completions request used upstream. */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });

  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls = content.flatMap((block) => {
        if (!isRecord(block) || block.type !== 'tool_use') return [];
        return [{
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        }];
      });
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text === '' ? null : text,
        ...(toolCalls.length === 0 ? {} : { tool_calls: toolCalls }),
      });
      continue;
    }

    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
      } else {
        const part = anthropicBlockToOpenAI(block);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages, max_tokens: numberOrZero(body.max_tokens) || 1024 };
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  if (isRecord(body.tool_choice)) {
    if (body.tool_choice.type === 'auto') out.tool_choice = 'auto';
    else if (body.tool_choice.type === 'any') out.tool_choice = 'required';
    else if (body.tool_choice.type === 'tool' && typeof body.tool_choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: body.tool_choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** Convert an OpenAI Chat Completions request into Anthropic Messages fields. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system' || raw.role === 'developer') {
      const text = textOf(raw.content);
      system = system === '' ? text : `${system}\n${text}`;
    } else if (raw.role === 'assistant') {
      messages.push(assistantOpenAIMessageToAnthropic(raw));
    } else if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{
          type: 'tool_result',
          tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
          content: toolResultText(raw.content),
        }],
      });
    } else {
      const content = openAIContentToAnthropic(raw.content);
      if (content.length > 0) messages.push({ role: 'user', content });
    }
  }

  const out: Json = {
    model,
    messages,
    max_tokens: numberOrZero(body.max_tokens ?? body.max_completion_tokens) || 1024,
  };
  if (system !== '') out.system = system;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop)) out.stop_sequences = body.stop;

  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  if (typeof body.tool_choice === 'string') {
    if (body.tool_choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (body.tool_choice === 'required') out.tool_choice = { type: 'any' };
  } else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function)) {
    const name = body.tool_choice.function.name;
    if (typeof name === 'string') out.tool_choice = { type: 'tool', name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function finishReasonOf(stop: unknown): string | null {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return stop === 'end_turn' || stop === 'stop_sequence' ? 'stop' : null;
}

function anthropicUsageToOpenAI(usage: Json): Json {
  const converted: Json = {
    prompt_tokens: numberOrZero(usage.input_tokens),
    completion_tokens: numberOrZero(usage.output_tokens),
    total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
  };
  if (typeof usage.cache_read_input_tokens === 'number') {
    converted.prompt_tokens_details = { cached_tokens: usage.cache_read_input_tokens };
  }
  return converted;
}

/** Convert an Anthropic Messages response to an OpenAI Chat Completions response. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const blocks = Array.isArray(body.content) ? body.content : [];
  const text: string[] = [];
  const toolCalls: Json[] = [];
  for (const raw of blocks) {
    if (!isRecord(raw)) continue;
    if (raw.type === 'text' && typeof raw.text === 'string') text.push(raw.text);
    if (raw.type === 'tool_use') {
      toolCalls.push({
        id: typeof raw.id === 'string' ? raw.id : '',
        type: 'function',
        function: {
          name: typeof raw.name === 'string' ? raw.name : '',
          arguments: JSON.stringify(isRecord(raw.input) ? raw.input : {}),
        },
      });
    }
  }
  const message: Json = { role: 'assistant', content: text.length === 0 ? null : text.join('') };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  const usage = isRecord(body.usage) ? anthropicUsageToOpenAI(body.usage) : anthropicUsageToOpenAI({});
  return {
    id: typeof body.id === 'string' ? body.id : 'chatcmpl_aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{
      index: 0,
      message,
      finish_reason: finishReasonOf(body.stop_reason),
    }],
    usage,
  };
}

/** Convert an OpenAI Chat Completions response to an Anthropic Messages response. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: objectOfJson(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });

  const usage = isRecord(body.usage) ? body.usage : {};
  const anthropicUsage: Json = {
    input_tokens: numberOrZero(usage.prompt_tokens),
    output_tokens: numberOrZero(usage.completion_tokens),
  };
  if (isRecord(usage.prompt_tokens_details) && typeof usage.prompt_tokens_details.cached_tokens === 'number') {
    anthropicUsage.cache_read_input_tokens = usage.prompt_tokens_details.cached_tokens;
  }
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: anthropicUsage,
  };
}

export function openAIJsonWithModel(body: Json, model: string): Json {
  return { ...body, model };
}

export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function dataLines(frame: string): string[] {
  return frame.split('\n').flatMap((line) => {
    const trimmed = line.trimStart();
    return trimmed.startsWith('data:') ? [trimmed.slice(5).trimStart()] : [];
  });
}

function frameWithModel(frame: string, model: string): string {
  const lines = frame.split('\n').map((line) => {
    const trimmed = line.trimStart();
    if (!trimmed.startsWith('data:')) return line;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    try {
      const value: unknown = JSON.parse(payload);
      if (isRecord(value)) return `data: ${JSON.stringify({ ...value, model })}`;
    } catch {
      // Preserve non-JSON data lines as sent.
    }
    return line;
  });
  return `${lines.join('\n')}\n\n`;
}

async function* sseFrames(upstream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n?/g, '\n');
      let end = buffer.indexOf('\n\n');
      while (end !== -1) {
        yield buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        end = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode().replace(/\r\n?/g, '\n');
    if (buffer.trim() !== '') yield buffer;
  } finally {
    reader.releaseLock();
  }
}

/** Relay OpenAI SSE, replacing the provider's model spelling on every JSON chunk. */
export async function* openAIStreamWithModel(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  for await (const frame of sseFrames(upstream)) yield frameWithModel(frame, model);
}

function anthropicEventWithModel(event: string, model: string): string {
  const lines = event.split('\n').map((line) => {
    if (!line.startsWith('data:')) return line;
    const payload = line.slice(5).trimStart();
    try {
      const value: unknown = JSON.parse(payload);
      if (!isRecord(value)) return line;
      const answer: Json = { ...value, model };
      if (value.type === 'message_start' && isRecord(value.message)) {
        answer.message = { ...value.message, model };
      }
      return `data: ${JSON.stringify(answer)}`;
    } catch {
      return line;
    }
  });
  return `${lines.join('\n')}\n\n`;
}

/** Translate OpenAI streaming completion chunks into Anthropic Messages events. */
export async function* anthropicEventsFromOpenAI(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let stopReason: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  let nextIndex = 0;
  let textIndex = -1;
  const toolIndexes = new Map<number, number>();

  const emit = (event: string, value: Json): string => anthropicEventWithModel(sseEvent(event, value).trimEnd(), model);
  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [emit('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: 0 },
      },
    })];
  };
  const closeBlocks = (): string[] => {
    const events: string[] = [];
    if (textIndex !== -1) {
      events.push(emit('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((left, right) => left - right)) {
      events.push(emit('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
    return events;
  };
  const finish = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeBlocks(),
      emit('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason ?? 'end_turn', stop_sequence: null },
        usage: { input_tokens: inputTokens, output_tokens: outputTokens },
      }),
      emit('message_stop', { type: 'message_stop' }),
    ];
  };

  const process = (frame: string): string[] => {
    const events: string[] = [];
    for (const payload of dataLines(frame)) {
      if (payload === '[DONE]') {
        events.push(...start(), ...finish());
        continue;
      }
      let parsed: unknown;
      try {
        parsed = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(parsed)) continue;
      if (isRecord(parsed.usage)) {
        inputTokens = numberOrZero(parsed.usage.prompt_tokens);
        outputTokens = numberOrZero(parsed.usage.completion_tokens);
      }
      const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      events.push(...start());

      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex++;
          events.push(emit('content_block_start', {
            type: 'content_block_start',
            index: textIndex,
            content_block: { type: 'text', text: '' },
          }));
        }
        events.push(emit('content_block_delta', {
          type: 'content_block_delta',
          index: textIndex,
          delta: { type: 'text_delta', text: delta.content },
        }));
      }

      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const sourceIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let blockIndex = toolIndexes.get(sourceIndex);
        if (blockIndex === undefined) {
          if (textIndex !== -1) {
            events.push(emit('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          blockIndex = nextIndex++;
          toolIndexes.set(sourceIndex, blockIndex);
          const fn = isRecord(call.function) ? call.function : {};
          events.push(emit('content_block_start', {
            type: 'content_block_start',
            index: blockIndex,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${blockIndex}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          }));
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          events.push(emit('content_block_delta', {
            type: 'content_block_delta',
            index: blockIndex,
            delta: { type: 'input_json_delta', partial_json: fn.arguments },
          }));
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
        stopReason = stopReasonOf(choice.finish_reason);
      }
    }
    return events;
  };

  for await (const frame of sseFrames(upstream)) {
    for (const event of process(frame)) yield event;
  }
  for (const event of start()) yield event;
  for (const event of finish()) yield event;
}

function anthropicFrameEvent(frame: string): string {
  for (const line of frame.split('\n')) {
    const trimmed = line.trimStart();
    if (trimmed.startsWith('event:')) return trimmed.slice(6).trim();
  }
  return '';
}

/** Translate Anthropic Messages events into OpenAI chunks, including tool argument deltas and usage. */
export async function* openAIStreamFromAnthropic(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  let id = 'chatcmpl_stream';
  let created = FIXED_OPENAI_CREATED;
  let promptTokens = 0;
  let completionTokens = 0;
  const toolIndexes = new Map<number, number>();
  let nextToolIndex = 0;
  let done = false;
  const chunk = (choices: Json[], usage?: Json): string => sseEvent(null, {
    id,
    object: 'chat.completion.chunk',
    created,
    model,
    choices,
    ...(usage === undefined ? {} : { usage }),
  });

  for await (const frame of sseFrames(upstream)) {
    const eventName = anthropicFrameEvent(frame);
    const payload = dataLines(frame).join('\n');
    if (payload === '[DONE]') {
      yield 'data: [DONE]\n\n';
      done = true;
      continue;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(parsed)) continue;
    if (eventName === 'message_start' && isRecord(parsed.message)) {
      const message = parsed.message;
      if (typeof message.id === 'string') id = message.id;
      if (isRecord(message.usage)) promptTokens = numberOrZero(message.usage.input_tokens);
      yield chunk([{ index: 0, delta: { role: 'assistant' }, finish_reason: null }]);
    } else if (eventName === 'content_block_start' && isRecord(parsed.content_block)) {
      const block = parsed.content_block;
      if (block.type === 'text') {
        if (typeof block.text === 'string' && block.text !== '') {
          yield chunk([{ index: 0, delta: { content: block.text }, finish_reason: null }]);
        }
      } else if (block.type === 'tool_use' && typeof parsed.index === 'number') {
        const toolIndex = nextToolIndex++;
        toolIndexes.set(parsed.index, toolIndex);
        yield chunk([{
          index: 0,
          delta: {
            tool_calls: [{
              index: toolIndex,
              id: typeof block.id === 'string' ? block.id : `call_${toolIndex}`,
              type: 'function',
              function: { name: typeof block.name === 'string' ? block.name : '', arguments: '' },
            }],
          },
          finish_reason: null,
        }]);
      }
    } else if (eventName === 'content_block_delta' && isRecord(parsed.delta)) {
      const delta = parsed.delta;
      if (delta.type === 'text_delta' && typeof delta.text === 'string') {
        yield chunk([{ index: 0, delta: { content: delta.text }, finish_reason: null }]);
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
        const index = typeof parsed.index === 'number' ? toolIndexes.get(parsed.index) : undefined;
        if (index !== undefined) {
          yield chunk([{
            index: 0,
            delta: {
              tool_calls: [{ index, function: { arguments: delta.partial_json } }],
            },
            finish_reason: null,
          }]);
        }
      }
    } else if (eventName === 'message_delta') {
      const delta = isRecord(parsed.delta) ? parsed.delta : {};
      const usage = isRecord(parsed.usage) ? parsed.usage : {};
      completionTokens = numberOrZero(usage.output_tokens);
      yield chunk(
        [{ index: 0, delta: {}, finish_reason: finishReasonOf(delta.stop_reason) }],
        {
          prompt_tokens: promptTokens,
          completion_tokens: completionTokens,
          total_tokens: promptTokens + completionTokens,
        },
      );
    } else if (eventName === 'message_stop' && !done) {
      yield 'data: [DONE]\n\n';
      done = true;
    }
  }
  if (!done) yield 'data: [DONE]\n\n';
}
