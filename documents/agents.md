Agents
Agents are the main building‑block of the OpenAI Agents SDK. An Agent is a Large Language Model (LLM) that has been configured with:

Instructions – the system prompt that tells the model who it is and how it should respond.
Model – which OpenAI model to call, plus any optional model tuning parameters.
Tools – a list of functions or APIs the LLM can invoke to accomplish a task.
Basic Agent definition
import { Agent } from '@openai/agents';

const agent = new Agent({
  name: 'Haiku Agent',
  instructions: 'Always respond in haiku form.',
  model: 'o4-mini', // optional – falls back to the default model
});

The rest of this page walks through every Agent feature in more detail.

Basic configuration
The Agent constructor takes a single configuration object. The most commonly‑used properties are shown below.

Property	Required	Description
name	yes	A short human‑readable identifier.
instructions	yes	System prompt (string or function – see Dynamic instructions).
model	no	Model name or a custom Model implementation.
modelSettings	no	Tuning parameters (temperature, top_p, etc.).
tools	no	Array of Tool instances the model can call.
Agent with tools
import { Agent, tool } from '@openai/agents';
import { z } from 'zod';

const getWeather = tool({
  name: 'get_weather',
  description: 'Return the weather for a given city.',
  parameters: z.object({ city: z.string() }),
  async execute({ city }) {
    return `The weather in ${city} is sunny.`;
  },
});

const agent = new Agent({
  name: 'Weather bot',
  instructions: 'You are a helpful weather bot.',
  model: 'o4-mini',
  tools: [getWeather],
});

Context
Agents are generic on their context type – i.e. Agent<TContext, TOutput>. The context is a dependency‑injection object that you create and pass to Runner.run(). It is forwarded to every tool, guardrail, handoff, etc. and is useful for storing state or providing shared services (database connections, user metadata, feature flags, …).

Agent with context
import { Agent } from '@openai/agents';

interface Purchase {
  id: string;
  uid: string;
  deliveryStatus: string;
}
interface UserContext {
  uid: string;
  isProUser: boolean;

  // this function can be used within tools
  fetchPurchases(): Promise<Purchase[]>;
}

const agent = new Agent<UserContext>({
  name: 'Personal shopper',
  instructions: 'Recommend products the user will love.',
});

// Later
import { run } from '@openai/agents';

const result = await run(agent, 'Find me a new pair of running shoes', {
  context: { uid: 'abc', isProUser: true, fetchPurchases: async () => [] },
});

Output types
By default, an Agent returns plain text (string). If you want the model to return a structured object you can specify the outputType property. The SDK accepts:

A Zod schema (z.object({...})).
Any JSON‑schema‑compatible object.
Structured output with Zod
import { Agent } from '@openai/agents';
import { z } from 'zod';

const CalendarEvent = z.object({
  name: z.string(),
  date: z.string(),
  participants: z.array(z.string()),
});

const extractor = new Agent({
  name: 'Calendar extractor',
  instructions: 'Extract calendar events from the supplied text.',
  outputType: CalendarEvent,
});

When outputType is provided, the SDK automatically uses structured outputs instead of plain text.

Handoffs
An Agent can delegate to other Agents via the handoffs property. A common pattern is to use a triage agent that routes the conversation to a more specialised sub‑agent.

Agent with handoffs
import { Agent } from '@openai/agents';

const bookingAgent = new Agent({
  name: 'Booking Agent',
  instructions: 'Help users with booking requests.',
});

const refundAgent = new Agent({
  name: 'Refund Agent',
  instructions: 'Process refund requests politely and efficiently.',
});

// Use Agent.create method to ensure the finalOutput type considers handoffs
const triageAgent = Agent.create({
  name: 'Triage Agent',
  instructions: [
    'Help the user with their questions.',
    'If the user asks about booking, hand off to the booking agent.',
    'If the user asks about refunds, hand off to the refund agent.',
  ].join('\n'),
  handoffs: [bookingAgent, refundAgent],
});

You can read more about this pattern in the handoffs guide.

Dynamic instructions
instructions can be a function instead of a string. The function receives the current RunContext and the Agent instance and can return a string or a Promise<string>.

Agent with dynamic instructions
import { Agent, RunContext } from '@openai/agents';

interface UserContext {
  name: string;
}

function buildInstructions(runContext: RunContext<UserContext>) {
  return `The user's name is ${runContext.context.name}.  Be extra friendly!`;
}

const agent = new Agent<UserContext>({
  name: 'Personalized helper',
  instructions: buildInstructions,
});

Both synchronous and async functions are supported.

Lifecycle hooks
For advanced use‑cases you can observe the Agent lifecycle by listening on events

Agent with lifecycle hooks
import { Agent } from '@openai/agents';

const agent = new Agent({
  name: 'Verbose agent',
  instructions: 'Explain things thoroughly.',
});

agent.on('agent_start', (ctx, agent) => {
  console.log(`[${agent.name}] started`);
});
agent.on('agent_end', (ctx, output) => {
  console.log(`[agent] produced:`, output);
});

Guardrails
Guardrails allow you to validate or transform user input and agent output. They are configured via the inputGuardrails and outputGuardrails arrays. See the guardrails guide for details.

Cloning / copying agents
Need a slightly modified version of an existing agent? Use the clone() method, which returns an entirely new Agent instance.

Cloning Agents
import { Agent } from '@openai/agents';

const pirateAgent = new Agent({
  name: 'Pirate',
  instructions: 'Respond like a pirate – lots of “Arrr!”',
  model: 'o4-mini',
});

const robotAgent = pirateAgent.clone({
  name: 'Robot',
  instructions: 'Respond like a robot – be precise and factual.',
});

Forcing tool use
Supplying tools doesn’t guarantee the LLM will call one. You can force tool use with modelSettings.tool_choice:

'auto' (default) – the LLM decides whether to use a tool.
'required' – the LLM must call a tool (it can choose which one).
'none' – the LLM must not call a tool.
A specific tool name, e.g. 'calculator' – the LLM must call that particular tool.
Forcing tool use
import { Agent, tool } from '@openai/agents';
import { z } from 'zod';

const calculatorTool = tool({
  name: 'Calculator',
  description: 'Use this tool to answer questions about math problems.',
  parameters: z.object({ question: z.string() }),
  execute: async (input) => {
    throw new Error('TODO: implement this');
  },
});

const agent = new Agent({
  name: 'Strict tool user',
  instructions: 'Always answer using the calculator tool.',
  tools: [calculatorTool],
  modelSettings: { toolChoice: 'auto' },
});

Preventing infinite loops
After a tool call the SDK automatically resets tool_choice back to 'auto'. This prevents the model from entering an infinite loop where it repeatedly tries to call the tool. You can override this behaviour via the resetToolChoice flag or by configuring toolUseBehavior:

'run_llm_again' (default) – run the LLM again with the tool result.
'stop_on_first_tool' – treat the first tool result as the final answer.
{ stopAtToolNames: ['my_tool'] } – stop when any of the listed tools is called.
(context, toolResults) => ... – custom function returning whether the run should finish.
const agent = new Agent({
  ...,
  toolUseBehavior: 'stop_on_first_tool',
});

Running agents
Agents do nothing by themselves – you run them with the Runner class or the run() utility.

Simple run
import { Agent, run } from '@openai/agents';

const agent = new Agent({
  name: 'Assistant',
  instructions: 'You are a helpful assistant',
});

const result = await run(
  agent,
  'Write a haiku about recursion in programming.',
);
console.log(result.finalOutput);

// Code within the code,
// Functions calling themselves,
// Infinite loop's dance.

When you don’t need a custom runner, you can also use the run() utility, which runs a singletone default Runner instance.

Alternatively, you can create your own runner instance:

Simple run
import { Agent, Runner } from '@openai/agents';

const agent = new Agent({
  name: 'Assistant',
  instructions: 'You are a helpful assistant',
});

// You can pass custom configuration to the runner
const runner = new Runner();

const result = await runner.run(
  agent,
  'Write a haiku about recursion in programming.',
);
console.log(result.finalOutput);

// Code within the code,
// Functions calling themselves,
// Infinite loop's dance.

After running your agent, you will receive a result object that contains the final output and the full history of the run.

The agent loop
When you use the run method in Runner, you pass in a starting agent and input. The input can either be a string (which is considered a user message), or a list of input items, which are the items in the OpenAI Responses API.

The runner then runs a loop:

Call the current agent’s model with the current input.
Inspect the LLM response.
Final output → return.
Handoff → switch to the new agent, keep the accumulated conversation history, go to 1.
Tool calls → execute tools, append their results to the conversation, go to 1.
Throw MaxTurnsExceededError once maxTurns is reached.
Note

The rule for whether the LLM output is considered as a “final output” is that it produces text output with the desired type, and there are no tool calls.

Runner lifecycle
Create a Runner when your app starts and reuse it across requests. The instance stores global configuration such as model provider and tracing options. Only create another Runner if you need a completely different setup. For simple scripts you can also call run() which uses a default runner internally.

Run arguments
The input to the run() method is an initial agent to start the run on, input for the run and a set of options.

The input can either be a string (which is considered a user message), or a list of input items, or a RunState object in case you are building a human-in-the-loop agent.

The additional options are:

Option	Default	Description
stream	false	If true the call returns a StreamedRunResult and emits events as they arrive from the model.
context	–	Context object forwarded to every tool / guardrail / handoff. Learn more in the context guide.
maxTurns	10	Safety limit – throws MaxTurnsExceededError when reached.
signal	–	AbortSignal for cancellation.
Streaming
Streaming allows you to additionally receive streaming events as the LLM runs. Once the stream is started, the StreamedRunResult will contain the complete information about the run, including all the new outputs produces. You can iterate over the streaming events using a for await loop. Read more in the streaming guide.

Run config
If you are creating your own Runner instance, you can pass in a RunConfig object to configure the runner.

Field	Type	Purpose
model	string | Model	Force a specific model for all agents in the run.
modelProvider	ModelProvider	Resolves model names – defaults to the OpenAI provider.
modelSettings	ModelSettings	Global tuning parameters that override per‑agent settings.
handoffInputFilter	HandoffInputFilter	Mutates input items when performing handoffs (if the handoff itself doesn’t already define one).
inputGuardrails	InputGuardrail[]	Guardrails applied to the initial user input.
outputGuardrails	OutputGuardrail[]	Guardrails applied to the final output.
tracingDisabled	boolean	Disable OpenAI Tracing completely.
traceIncludeSensitiveData	boolean	Exclude LLM/tool inputs & outputs from traces while still emitting spans.
workflowName	string	Appears in the Traces dashboard – helps group related runs.
traceId / groupId	string	Manually specify the trace or group ID instead of letting the SDK generate one.
traceMetadata	Record<string, any>	Arbitrary metadata to attach to every span.
Conversations / chat threads
Each call to runner.run() (or run() utility) represents one turn in your application-level conversation. You choose how much of the RunResult you show the end‑user – sometimes only finalOutput, other times every generated item.

Example of carrying over the conversation history
import { Agent, AgentInputItem, run } from '@openai/agents';

let thread: AgentInputItem[] = [];

const agent = new Agent({
  name: 'Assistant',
});

async function userSays(text: string) {
  const result = await run(
    agent,
    thread.concat({ role: 'user', content: text }),
  );

  thread = result.history; // Carry over history + newly generated items
  return result.finalOutput;
}

await userSays('What city is the Golden Gate Bridge in?');
// -> "San Francisco"

await userSays('What state is it in?');
// -> "California"

See the chat example for an interactive version.

Exceptions
The SDK throws a small set of errors you can catch:

MaxTurnsExceededError – maxTurns reached.
ModelBehaviorError – model produced invalid output (e.g. malformed JSON, unknown tool).
InputGuardrailTripwireTriggered / OutputGuardrailTripwireTriggered – guardrail violations.
GuardrailExecutionError – guardrails failed to complete.
ToolCallError – any of function tool calls failed.
UserError – any error thrown based on configuration or user input.
All extend the base AgentsError class, which could provide the state property to access the current run state.

Here is an example code that handles GuardrailExecutionError:

Guardrail execution error
import {
  Agent,
  run,
  GuardrailExecutionError,
  InputGuardrail,
  InputGuardrailTripwireTriggered,
} from '@openai/agents';
import { z } from 'zod';

const guardrailAgent = new Agent({
  name: 'Guardrail check',
  instructions: 'Check if the user is asking you to do their math homework.',
  outputType: z.object({
    isMathHomework: z.boolean(),
    reasoning: z.string(),
  }),
});

const unstableGuardrail: InputGuardrail = {
  name: 'Math Homework Guardrail (unstable)',
  execute: async () => {
    throw new Error('Something is wrong!');
  },
};

const fallbackGuardrail: InputGuardrail = {
  name: 'Math Homework Guardrail (fallback)',
  execute: async ({ input, context }) => {
    const result = await run(guardrailAgent, input, { context });
    return {
      outputInfo: result.finalOutput,
      tripwireTriggered: result.finalOutput?.isMathHomework ?? false,
    };
  },
};

const agent = new Agent({
  name: 'Customer support agent',
  instructions:
    'You are a customer support agent. You help customers with their questions.',
  inputGuardrails: [unstableGuardrail],
});

async function main() {
  try {
    const input = 'Hello, can you help me solve for x: 2x + 3 = 11?';
    const result = await run(agent, input);
    console.log(result.finalOutput);
  } catch (e) {
    if (e instanceof GuardrailExecutionError) {
      console.error(`Guardrail execution failed: ${e}`);
      // If you want to retry the execution with different settings,
      // you can reuse the runner's latest state this way:
      if (e.state) {
        try {
          agent.inputGuardrails = [fallbackGuardrail]; // fallback
          const result = await run(agent, e.state);
          console.log(result.finalOutput);
        } catch (ee) {
          if (ee instanceof InputGuardrailTripwireTriggered) {
            console.log('Math homework guardrail tripped');
          }
        }
      }
    } else {
      throw e;
    }
  }
}

main().catch(console.error);

When you run the above example, you will see the following output:

Guardrail execution failed: Error: Input guardrail failed to complete: Error: Something is wrong!
Math homework guardrail tripped

Results
When you run your agent, you will either receive a:

RunResult if you call run without stream: true
StreamedRunResult if you call run with stream: true. For details on streaming, also check the streaming guide.
Final output
The finalOutput property contains the final output of the last agent that ran. This result is either:

string — default for any agent that has no outputType defined
unknown — if the agent has a JSON schema defined as output type. In this case the JSON was parsed but you still have to verify its type manually.
z.infer<outputType> — if the agent has a Zod schema defined as output type. The output will automatically be parsed against this schema.
undefined — if the agent did not produce an output (for example stopped before it could produce an output)
If you are using handoffs with different output types, you should use the Agent.create() method instead of the new Agent() constructor to create your agents.

This will enable the SDK to infer the output types across all possible handoffs and provide a union type for the finalOutput property.

For example:

Handoff final output types
import { Agent, run } from '@openai/agents';
import { z } from 'zod';

const refundAgent = new Agent({
  name: 'Refund Agent',
  instructions:
    'You are a refund agent. You are responsible for refunding customers.',
  outputType: z.object({
    refundApproved: z.boolean(),
  }),
});

const orderAgent = new Agent({
  name: 'Order Agent',
  instructions:
    'You are an order agent. You are responsible for processing orders.',
  outputType: z.object({
    orderId: z.string(),
  }),
});

const triageAgent = Agent.create({
  name: 'Triage Agent',
  instructions:
    'You are a triage agent. You are responsible for triaging customer issues.',
  handoffs: [refundAgent, orderAgent],
});

const result = await run(triageAgent, 'I need to a refund for my order');

const output = result.finalOutput;
// ^? { refundApproved: boolean } | { orderId: string } | string | undefined

Inputs for the next turn
There are two ways you can access the inputs for your next turn:

result.history — contains a copy of both your input and the output of the agents.
result.output — contains the output of the full agent run.
history is a convenient way to maintain a full history in a chat-like use case:

History loop
import { AgentInputItem, Agent, user, run } from '@openai/agents';

const agent = new Agent({
  name: 'Assistant',
  instructions:
    'You are a helpful assistant knowledgeable about recent AGI research.',
});

let history: AgentInputItem[] = [
  // intial message
  user('Are we there yet?'),
];

for (let i = 0; i < 10; i++) {
  // run 10 times
  const result = await run(agent, history);

  // update the history to the new output
  history = result.history;

  history.push(user('How about now?'));
}

Last agent
The lastAgent property contains the last agent that ran. Depending on your application, this is often useful for the next time the user inputs something. For example, if you have a frontline triage agent that hands off to a language-specific agent, you can store the last agent, and re-use it the next time the user messages the agent.

In streaming mode it can also be useful to access the currentAgent property that’s mapping to the current agent that is running.

New items
The newItems property contains the new items generated during the run. The items are RunItems. A run item wraps the raw item generated by the LLM. These can be used to access additionally to the output of the LLM which agent these events were associated with.

RunMessageOutputItem indicates a message from the LLM. The raw item is the message generated.
RunHandoffCallItem indicates that the LLM called the handoff tool. The raw item is the tool call item from the LLM.
RunHandoffOutputItem indicates that a handoff occurred. The raw item is the tool response to the handoff tool call. You can also access the source/target agents from the item.
RunToolCallItem indicates that the LLM invoked a tool.
RunToolCallOutputItem indicates that a tool was called. The raw item is the tool response. You can also access the tool output from the item.
RunReasoningItem indicates a reasoning item from the LLM. The raw item is the reasoning generated.
RunToolApprovalItem indicates that the LLM requested approval for a tool call. The raw item is the tool call item from the LLM.
State
The state property contains the state of the run. Most of what is attached to the result is derived from the state but the state is serializable/deserializable and can also be used as input for a subsequent call to run in case you need to recover from an error or deal with an interruption.

Interruptions
If you are using needsApproval in your agent, your run might trigger some interruptions that you need to handle before continuing. In that case interruptions will be an array of ToolApprovalItems that caused the interruption. Check out the human-in-the-loop guide for more information on how to work with interruptions.

Other information
Raw responses
The rawResponses property contains the raw LLM responses generated by the model during the agent run.

Last response ID
The lastResponseId property contains the ID of the last response generated by the model during the agent run.

Guardrail results
The inputGuardrailResults and outputGuardrailResults properties contain the results of the guardrails, if any. Guardrail results can sometimes contain useful information you want to log or store, so we make these available to you.

Original input
The input property contains the original input you provided to the run method. In most cases you won’t need this, but it’s available in case you do.

Tools
Tools let an Agent take actions – fetch data, call external APIs, execute code, or even use a computer. The JavaScript/TypeScript SDK supports four categories:

Hosted tools – run alongside the model on OpenAI servers. (web search, file search, computer use, code interpreter, image generation)
Function tools – wrap any local function with a JSON schema so the LLM can call it.
Agents as tools – expose an entire Agent as a callable tool.
Local MCP servers – attach a Model Context Protocol server running on your machine.
1. Hosted tools
When you use the OpenAIResponsesModel you can add the following built‑in tools:

Tool	Type string	Purpose
Web search	'web_search'	Internet search.
File / retrieval search	'file_search'	Query vector stores hosted on OpenAI.
Computer use	'computer'	Automate GUI interactions.
Code Interpreter	'code_interpreter'	Run code in a sandboxed environment.
Image generation	'image_generation'	Generate images based on text.
Hosted tools
import { Agent, webSearchTool, fileSearchTool } from '@openai/agents';

const agent = new Agent({
  name: 'Travel assistant',
  tools: [webSearchTool(), fileSearchTool('VS_ID')],
});

The exact parameter sets match the OpenAI Responses API – refer to the official documentation for advanced options like rankingOptions or semantic filters.

2. Function tools
You can turn any function into a tool with the tool() helper.

Function tool with Zod parameters
import { tool } from '@openai/agents';
import { z } from 'zod';

const getWeatherTool = tool({
  name: 'get_weather',
  description: 'Get the weather for a given city',
  parameters: z.object({ city: z.string() }),
  async execute({ city }) {
    return `The weather in ${city} is sunny.`;
  },
});

Options reference
Field	Required	Description
name	No	Defaults to the function name (e.g., get_weather).
description	Yes	Clear, human-readable description shown to the LLM.
parameters	Yes	Either a Zod schema or a raw JSON schema object. Zod parameters automatically enable strict mode.
strict	No	When true (default), the SDK returns a model error if the arguments don’t validate. Set to false for fuzzy matching.
execute	Yes	(args, context) => string | Promise<string>– your business logic. The optional second parameter is theRunContext.
errorFunction	No	Custom handler (context, error) => string for transforming internal errors into a user-visible string.
Non‑strict JSON‑schema tools
If you need the model to guess invalid or partial input you can disable strict mode when using raw JSON schema:

Non-strict JSON schema tools
import { tool } from '@openai/agents';

interface LooseToolInput {
  text: string;
}

const looseTool = tool({
  description: 'Echo input; be forgiving about typos',
  strict: false,
  parameters: {
    type: 'object',
    properties: { text: { type: 'string' } },
    required: ['text'],
    additionalProperties: true,
  },
  execute: async (input) => {
    // because strict is false we need to do our own verification
    if (typeof input !== 'object' || input === null || !('text' in input)) {
      return 'Invalid input. Please try again';
    }
    return (input as LooseToolInput).text;
  },
});

3. Agents as tools
Sometimes you want an Agent to assist another Agent without fully handing off the conversation. Use agent.asTool():

Agents as tools
import { Agent } from '@openai/agents';

const summarizer = new Agent({
  name: 'Summarizer',
  instructions: 'Generate a concise summary of the supplied text.',
});

const summarizerTool = summarizer.asTool({
  toolName: 'summarize_text',
  toolDescription: 'Generate a concise summary of the supplied text.',
});

const mainAgent = new Agent({
  name: 'Research assistant',
  tools: [summarizerTool],
});

Under the hood the SDK:

Creates a function tool with a single input parameter.
Runs the sub‑agent with that input when the tool is called.
Returns either the last message or the output extracted by customOutputExtractor.
4. Local MCP servers
You can expose tools via a local Model Context Protocol server and attach them to an agent. Use MCPServerStdio to spawn and connect to the server:

Local MCP server
import { Agent, MCPServerStdio } from '@openai/agents';

const server = new MCPServerStdio({
  fullCommand: 'npx -y @modelcontextprotocol/server-filesystem ./sample_files',
});

await server.connect();

const agent = new Agent({
  name: 'Assistant',
  mcpServers: [server],
});

See filesystem-example.ts for a complete example.

Tool use behaviour
Refer to the Agents guide for controlling when and how a model must use tools (tool_choice, toolUseBehavior, etc.).

Best practices
Short, explicit descriptions – describe what the tool does and when to use it.
Validate inputs – use Zod schemas for strict JSON validation where possible.
Avoid side‑effects in error handlers – errorFunction should return a helpful string, not throw.
One responsibility per tool – small, composable tools lead to better model reasoning.

Orchestrating multiple agents
Orchestration refers to the flow of agents in your app. Which agents run, in what order, and how do they decide what happens next? There are two main ways to orchestrate agents:

Allowing the LLM to make decisions: this uses the intelligence of an LLM to plan, reason, and decide on what steps to take based on that.
Orchestrating via code: determining the flow of agents via your code.
You can mix and match these patterns. Each has their own tradeoffs, described below.

Orchestrating via LLM
An agent is an LLM equipped with instructions, tools and handoffs. This means that given an open-ended task, the LLM can autonomously plan how it will tackle the task, using tools to take actions and acquire data, and using handoffs to delegate tasks to sub-agents. For example, a research agent could be equipped with tools like:

Web search to find information online
File search and retrieval to search through proprietary data and connections
Computer use to take actions on a computer
Code execution to do data analysis
Handoffs to specialized agents that are great at planning, report writing and more.
This pattern is great when the task is open-ended and you want to rely on the intelligence of an LLM. The most important tactics here are:

Invest in good prompts. Make it clear what tools are available, how to use them, and what parameters it must operate within.
Monitor your app and iterate on it. See where things go wrong, and iterate on your prompts.
Allow the agent to introspect and improve. For example, run it in a loop, and let it critique itself; or, provide error messages and let it improve.
Have specialized agents that excel in one task, rather than having a general purpose agent that is expected to be good at anything.
Invest in evals. This lets you train your agents to improve and get better at tasks.
Orchestrating via code
While orchestrating via LLM is powerful, orchestrating via code makes tasks more deterministic and predictable, in terms of speed, cost and performance. Common patterns here are:

Using structured outputs to generate well formed data that you can inspect with your code. For example, you might ask an agent to classify the task into a few categories, and then pick the next agent based on the category.
Chaining multiple agents by transforming the output of one into the input of the next. You can decompose a task like writing a blog post into a series of steps - do research, write an outline, write the blog post, critique it, and then improve it.
Running the agent that performs the task in a while loop with an agent that evaluates and provides feedback, until the evaluator says the output passes certain criteria.
Running multiple agents in parallel, e.g. via JavaScript primitives like Promise.all. This is useful for speed when you have multiple tasks that don’t depend on each other.
We have a number of examples in examples/agent-pa

Handoffs
Handoffs let an agent delegate part of a conversation to another agent. This is useful when different agents specialise in specific areas. In a customer support app for example, you might have agents that handle bookings, refunds or FAQs.

Handoffs are represented as tools to the LLM. If you hand off to an agent called Refund Agent, the tool name would be transfer_to_refund_agent.

Creating a handoff
Every agent accepts a handoffs option. It can contain other Agent instances or Handoff objects returned by the handoff() helper.

Basic usage
Basic handoffs
import { Agent, handoff } from '@openai/agents';

const billingAgent = new Agent({ name: 'Billing agent' });
const refundAgent = new Agent({ name: 'Refund agent' });

// Use Agent.create method to ensure the finalOutput type considers handoffs
const triageAgent = Agent.create({
  name: 'Triage agent',
  handoffs: [billingAgent, handoff(refundAgent)],
});

Customising handoffs via handoff()
The handoff() function lets you tweak the generated tool.

agent – the agent to hand off to.
toolNameOverride – override the default transfer_to_<agent_name> tool name.
toolDescriptionOverride – override the default tool description.
onHandoff – callback when the handoff occurs. Receives a RunContext and optionally parsed input.
inputType – expected input schema for the handoff.
inputFilter – filter the history passed to the next agent.
Customized handoffs
import { Agent, handoff, RunContext } from '@openai/agents';

function onHandoff(ctx: RunContext) {
  console.log('Handoff called');
}

const agent = new Agent({ name: 'My agent' });

const handoffObj = handoff(agent, {
  onHandoff,
  toolNameOverride: 'custom_handoff_tool',
  toolDescriptionOverride: 'Custom description',
});

Handoff inputs
Sometimes you want the LLM to provide data when invoking a handoff. Define an input schema and use it in handoff().

Handoff inputs
import { z } from 'zod';
import { Agent, handoff, RunContext } from '@openai/agents';

const EscalationData = z.object({ reason: z.string() });
type EscalationData = z.infer<typeof EscalationData>;

async function onHandoff(
  ctx: RunContext<EscalationData>,
  input: EscalationData | undefined,
) {
  console.log(`Escalation agent called with reason: ${input?.reason}`);
}

const agent = new Agent<EscalationData>({ name: 'Escalation agent' });

const handoffObj = handoff(agent, {
  onHandoff,
  inputType: EscalationData,
});

Input filters
By default a handoff receives the entire conversation history. To modify what gets passed to the next agent, provide an inputFilter. Common helpers live in @openai/agents-core/extensions.

Input filters
import { Agent, handoff } from '@openai/agents';
import { removeAllTools } from '@openai/agents-core/extensions';

const agent = new Agent({ name: 'FAQ agent' });

const handoffObj = handoff(agent, {
  inputFilter: removeAllTools,
});

Recommended prompts
LLMs respond more reliably when your prompts mention handoffs. The SDK exposes a recommended prefix via RECOMMENDED_PROMPT_PREFIX.

Recommended prompts
import { Agent } from '@openai/agents';
import { RECOMMENDED_PROMPT_PREFIX } from '@openai/agents-core/extensions';

const billingAgent = new Agent({
  name: 'Billing agent',
  instructions: `${RECOMMENDED_PROMPT_PREFIX}
Fill in the rest of your prompt here.`,
});

Context management
Context is an overloaded term. There are two main classes of context you might care about:

Local context that your code can access during a run: dependencies or data needed by tools, callbacks like onHandoff, and lifecycle hooks.
Agent/LLM context that the language model can see when generating a response.
Local context
Local context is represented by the RunContext<T> type. You create any object to hold your state or dependencies and pass it to Runner.run(). All tool calls and hooks receive a RunContext wrapper so they can read from or modify that object.

Local context example
import { Agent, run, RunContext, tool } from '@openai/agents';
import { z } from 'zod';

interface UserInfo {
  name: string;
  uid: number;
}

const fetchUserAge = tool({
  name: 'fetch_user_age',
  description: 'Return the age of the current user',
  parameters: z.object({}),
  execute: async (
    _args,
    runContext?: RunContext<UserInfo>,
  ): Promise<string> => {
    return `User ${runContext?.context.name} is 47 years old`;
  },
});

async function main() {
  const userInfo: UserInfo = { name: 'John', uid: 123 };

  const agent = new Agent<UserInfo>({
    name: 'Assistant',
    tools: [fetchUserAge],
  });

  const result = await run(agent, 'What is the age of the user?', {
    context: userInfo,
  });

  console.log(result.finalOutput);
  // The user John is 47 years old.
}

if (require.main === module) {
  main().catch(console.error);
}

Every agent, tool and hook participating in a single run must use the same type of context.

Use local context for things like:

Data about the run (user name, IDs, etc.)
Dependencies such as loggers or data fetchers
Helper functions
Note

The context object is not sent to the LLM. It is purely local and you can read from or write to it freely.

Agent/LLM context
When the LLM is called, the only data it can see comes from the conversation history. To make additional information available you have a few options:

Add it to the Agent instructions – also known as a system or developer message. This can be a static string or a function that receives the context and returns a string.
Include it in the input when calling Runner.run(). This is similar to the instructions technique but lets you place the message lower in the chain of command.
Expose it via function tools so the LLM can fetch data on demand.
Use retrieval or web search tools to ground responses in relevant data from files, databases, or the web.

Models
Every Agent ultimately calls an LLM. The SDK abstracts models behind two lightweight interfaces:

Model – knows how to make one request against a specific API.
ModelProvider – resolves human‑readable model names (e.g. 'gpt‑4o') to Model instances.
In day‑to‑day work you normally only interact with model names and occasionally ModelSettings.

Specifying a model per‑agent
import { Agent } from '@openai/agents';

const agent = new Agent({
  name: 'Creative writer',
  model: 'gpt-4.1',
});

The OpenAI provider
The default ModelProvider resolves names using the OpenAI APIs. It supports two distinct endpoints:

API	Usage	Call setOpenAIAPI()
Chat Completions	Standard chat & function calls	setOpenAIAPI('chat_completions')
Responses	New streaming‑first generative API (tool calls, flexible outputs)	setOpenAIAPI('responses') (default)
Authentication
Set default OpenAI key
import { setDefaultOpenAIKey } from '@openai/agents';

setDefaultOpenAIKey(process.env.OPENAI_API_KEY!); // sk-...

You can also plug your own OpenAI client via setDefaultOpenAIClient(client) if you need custom networking settings.

Default model
The OpenAI provider defaults to gpt‑4o. Override per agent or globally:

Set a default model
import { Runner } from '@openai/agents';

const runner = new Runner({ model: 'gpt‑4.1-mini' });

ModelSettings
ModelSettings mirrors the OpenAI parameters but is provider‑agnostic.

Field	Type	Notes
temperature	number	Creativity vs. determinism.
topP	number	Nucleus sampling.
frequencyPenalty	number	Penalise repeated tokens.
presencePenalty	number	Encourage new tokens.
toolChoice	'auto' | 'required' | 'none' | string	See forcing tool use.
parallelToolCalls	boolean	Allow parallel function calls where supported.
truncation	'auto' | 'disabled'	Token truncation strategy.
maxTokens	number	Maximum tokens in the response.
store	boolean	Persist the response for retrieval / RAG workflows.
Attach settings at either level:

Model settings
import { Runner, Agent } from '@openai/agents';

const agent = new Agent({
  name: 'Creative writer',
  // ...
  modelSettings: { temperature: 0.7, toolChoice: 'auto' },
});

// or globally
new Runner({ modelSettings: { temperature: 0.3 } });

Runner‑level settings override any conflicting per‑agent settings.

Prompt
Agents can be configured with a prompt parameter, indicating a server-stored prompt configuration that should be used to control the Agent’s behavior. Currently, this option is only supported when you use the OpenAI Responses API.

Field	Type	Notes
promptId	string	Unique identifier for a prompt.
version	string	Version of the prompt you wish to use.
variables	object	A key/value pair of variables to substitute into the prompt. Values can be strings or content input types like text, images, or files.
Agent with prompt
import { Agent, run } from '@openai/agents';

async function main() {
  const agent = new Agent({
    name: 'Assistant',
    prompt: {
      promptId: 'pmpt_684b3b772e648193b92404d7d0101d8a07f7a7903e519946',
      version: '1',
      variables: {
        poem_style: 'limerick',
      },
    },
  });

  const result = await run(agent, 'Write about unrequited love.');
  console.log(result.finalOutput);
}

if (require.main === module) {
  main().catch(console.error);
}

Any additional agent configuration, like tools or instructions, will override the values you may have configured in your stored prompt.

Custom model providers
Implementing your own provider is straightforward – implement ModelProvider and Model and pass the provider to the Runner constructor:

Minimal custom provider
import {
  ModelProvider,
  Model,
  ModelRequest,
  ModelResponse,
  ResponseStreamEvent,
} from '@openai/agents-core';

import { Agent, Runner } from '@openai/agents';

class EchoModel implements Model {
  name: string;
  constructor() {
    this.name = 'Echo';
  }
  async getResponse(request: ModelRequest): Promise<ModelResponse> {
    return {
      usage: {},
      output: [{ role: 'assistant', content: request.input as string }],
    } as any;
  }
  async *getStreamedResponse(
    _request: ModelRequest,
  ): AsyncIterable<ResponseStreamEvent> {
    yield {
      type: 'response.completed',
      response: { output: [], usage: {} },
    } as any;
  }
}

class EchoProvider implements ModelProvider {
  getModel(_modelName?: string): Promise<Model> | Model {
    return new EchoModel();
  }
}

const runner = new Runner({ modelProvider: new EchoProvider() });
console.log(runner.config.modelProvider.getModel());
const agent = new Agent({
  name: 'Test Agent',
  instructions: 'You are a helpful assistant.',
  model: new EchoModel(),
  modelSettings: { temperature: 0.7, toolChoice: 'auto' },
});
console.log(agent.model);

Tracing exporter
When using the OpenAI provider you can opt‑in to automatic trace export by providing your API key:

Tracing exporter
import { setTracingExportApiKey } from '@openai/agents';

setTracingExportApiKey('sk-...');

This sends traces to the OpenAI dashboard where you can inspect the complete execution graph of your workflow.

Guardrails
Guardrails run in parallel to your agents, allowing you to perform checks and validations on user input or agent output. For example, you may run a lightweight model as a guardrail before invoking an expensive model. If the guardrail detects malicious usage, it can trigger an error and stop the costly model from running.

There are two kinds of guardrails:

Input guardrails run on the initial user input.
Output guardrails run on the final agent output.
Input guardrails
Input guardrails run in three steps:

The guardrail receives the same input passed to the agent.
The guardrail function executes and returns a GuardrailFunctionOutput wrapped inside an InputGuardrailResult.
If tripwireTriggered is true, an InputGuardrailTripwireTriggered error is thrown.
Note Input guardrails are intended for user input, so they only run if the agent is the first agent in the workflow. Guardrails are configured on the agent itself because different agents often require different guardrails.

Output guardrails
Output guardrails follow the same pattern:

The guardrail receives the same input passed to the agent.
The guardrail function executes and returns a GuardrailFunctionOutput wrapped inside an OutputGuardrailResult.
If tripwireTriggered is true, an OutputGuardrailTripwireTriggered error is thrown.
Note Output guardrails only run if the agent is the last agent in the workflow. For realtime voice interactions see the voice agents guide.

Tripwires
When a guardrail fails, it signals this via a tripwire. As soon as a tripwire is triggered, the runner throws the corresponding error and halts execution.

Implementing a guardrail
A guardrail is simply a function that returns a GuardrailFunctionOutput. Below is a minimal example that checks whether the user is asking for math homework help by running another agent under the hood.

Input guardrail example
import {
  Agent,
  run,
  InputGuardrailTripwireTriggered,
  InputGuardrail,
} from '@openai/agents';
import { z } from 'zod';

const guardrailAgent = new Agent({
  name: 'Guardrail check',
  instructions: 'Check if the user is asking you to do their math homework.',
  outputType: z.object({
    isMathHomework: z.boolean(),
    reasoning: z.string(),
  }),
});

const mathGuardrail: InputGuardrail = {
  name: 'Math Homework Guardrail',
  execute: async ({ input, context }) => {
    const result = await run(guardrailAgent, input, { context });
    return {
      outputInfo: result.finalOutput,
      tripwireTriggered: result.finalOutput?.isMathHomework ?? false,
    };
  },
};

const agent = new Agent({
  name: 'Customer support agent',
  instructions:
    'You are a customer support agent. You help customers with their questions.',
  inputGuardrails: [mathGuardrail],
});

async function main() {
  try {
    await run(agent, 'Hello, can you help me solve for x: 2x + 3 = 11?');
    console.log("Guardrail didn't trip - this is unexpected");
  } catch (e) {
    if (e instanceof InputGuardrailTripwireTriggered) {
      console.log('Math homework guardrail tripped');
    }
  }
}

main().catch(console.error);

Output guardrails work the same way.

Output guardrail example
import {
  Agent,
  run,
  OutputGuardrailTripwireTriggered,
  OutputGuardrail,
} from '@openai/agents';
import { z } from 'zod';

// The output by the main agent
const MessageOutput = z.object({ response: z.string() });
type MessageOutput = z.infer<typeof MessageOutput>;

// The output by the math guardrail agent
const MathOutput = z.object({ reasoning: z.string(), isMath: z.boolean() });

// The guardrail agent
const guardrailAgent = new Agent({
  name: 'Guardrail check',
  instructions: 'Check if the output includes any math.',
  outputType: MathOutput,
});

// An output guardrail using an agent internally
const mathGuardrail: OutputGuardrail<typeof MessageOutput> = {
  name: 'Math Guardrail',
  async execute({ agentOutput, context }) {
    const result = await run(guardrailAgent, agentOutput.response, {
      context,
    });
    return {
      outputInfo: result.finalOutput,
      tripwireTriggered: result.finalOutput?.isMath ?? false,
    };
  },
};

const agent = new Agent({
  name: 'Support agent',
  instructions:
    'You are a user support agent. You help users with their questions.',
  outputGuardrails: [mathGuardrail],
  outputType: MessageOutput,
});

async function main() {
  try {
    const input = 'Hello, can you help me solve for x: 2x + 3 = 11?';
    await run(agent, input);
    console.log("Guardrail didn't trip - this is unexpected");
  } catch (e) {
    if (e instanceof OutputGuardrailTripwireTriggered) {
      console.log('Math output guardrail tripped');
    }
  }
}

main().catch(console.error);

guardrailAgent is used inside the guardrail functions.
The guardrail function receives the agent input or output and returns the result.
Extra information can be included in the guardrail result.
agent defines the actual workflow where guardrails are applied.

Streaming
The Agents SDK can deliver output from the model and other execution steps incrementally. Streaming keeps your UI responsive and avoids waiting for the entire final result before updating the user.

Enabling streaming
Pass a { stream: true } option to Runner.run() to obtain a streaming object rather than a full result:

Enabling streaming
import { Agent, run } from '@openai/agents';

const agent = new Agent({
  name: 'Storyteller',
  instructions:
    'You are a storyteller. You will be given a topic and you will tell a story about it.',
});

const result = await run(agent, 'Tell me a story about a cat.', {
  stream: true,
});

When streaming is enabled the returned stream implements the AsyncIterable interface. Each yielded event is an object describing what happened within the run. Most applications only want the model’s text though, so the stream provides helpers.

Get the text output
Call stream.toTextStream() to obtain a stream of the emitted text. When compatibleWithNodeStreams is true the return value is a regular Node.js Readable. We can pipe it directly into process.stdout or another destination.

Logging out the text as it arrives
import { Agent, run } from '@openai/agents';

const agent = new Agent({
  name: 'Storyteller',
  instructions:
    'You are a storyteller. You will be given a topic and you will tell a story about it.',
});

const result = await run(agent, 'Tell me a story about a cat.', {
  stream: true,
});

result
  .toTextStream({
    compatibleWithNodeStreams: true,
  })
  .pipe(process.stdout);

The promise stream.completed resolves once the run and all pending callbacks are completed. Always await it if you want to ensure there is no more output.

Listen to all events
You can use a for await loop to inspect each event as it arrives. Useful information includes low level model events, any agent switches and SDK specific run information:

Listening to all events
import { Agent, run } from '@openai/agents';

const agent = new Agent({
  name: 'Storyteller',
  instructions:
    'You are a storyteller. You will be given a topic and you will tell a story about it.',
});

const result = await run(agent, 'Tell me a story about a cat.', {
  stream: true,
});

for await (const event of result) {
  // these are the raw events from the model
  if (event.type === 'raw_model_stream_event') {
    console.log(`${event.type} %o`, event.data);
  }
  // agent updated events
  if (event.type == 'agent_updated_stream_event') {
    console.log(`${event.type} %s`, event.agent.name);
  }
  // Agent SDK specific events
  if (event.type === 'run_item_stream_event') {
    console.log(`${event.type} %o`, event.item);
  }
}

See the streamed example for a fully worked script that prints both the plain text stream and the raw event stream.

Human in the loop while streaming
Streaming is compatible with handoffs that pause execution (for example when a tool requires approval). The interruption field on the stream object exposes the interruptions, and you can continue execution by calling state.approve() or state.reject() for each of them. Executing again with { stream: true } resumes streaming output.

Handling human approval while streaming
import { Agent, run } from '@openai/agents';

const agent = new Agent({
  name: 'Storyteller',
  instructions:
    'You are a storyteller. You will be given a topic and you will tell a story about it.',
});

let stream = await run(
  agent,
  'What is the weather in San Francisco and Oakland?',
  { stream: true },
);
stream.toTextStream({ compatibleWithNodeStreams: true }).pipe(process.stdout);
await stream.completed;

while (stream.interruptions?.length) {
  console.log(
    'Human-in-the-loop: approval required for the following tool calls:',
  );
  const state = stream.state;
  for (const interruption of stream.interruptions) {
    const approved = confirm(
      `Agent ${interruption.agent.name} would like to use the tool ${interruption.rawItem.name} with "${interruption.rawItem.arguments}". Do you approve?`,
    );
    if (approved) {
      state.approve(interruption);
    } else {
      state.reject(interruption);
    }
  }

  // Resume execution with streaming output
  stream = await run(agent, state, { stream: true });
  const textStream = stream.toTextStream({ compatibleWithNodeStreams: true });
  textStream.pipe(process.stdout);
  await stream.completed;
}

A fuller example that interacts with the user is human-in-the-loop-stream.ts.

Tips
Remember to wait for stream.completed before exiting to ensure all output has been flushed.
The initial { stream: true } option only applies to the call where it is provided. If you re-run with a RunState you must specify the option again.
If your application only cares about the textual result prefer toTextStream() to avoid dealing with individual event objects.
With streaming and the event system you can integrate an agent into a chat interface, terminal application or any place where users benefit from incremental updates.

Human in the loop
This guide demonstrates how to use the built-in human-in-the-loop support in the SDK to pause and resume agent runs based on human intervention.

The primary use case for this right now is asking for approval for sensitive tool executions.

Approval requests
You can define a tool that requires approval by setting the needsApproval option to true or to an async function that returns a boolean.

Tool approval definition
import { tool } from '@openai/agents';
import z from 'zod';

const sensitiveTool = tool({
  name: 'cancelOrder',
  description: 'Cancel order',
  parameters: z.object({
    orderId: z.number(),
  }),
  // always requires approval
  needsApproval: true,
  execute: async ({ orderId }, args) => {
    // prepare order return
  },
});

const sendEmail = tool({
  name: 'sendEmail',
  description: 'Send an email',
  parameters: z.object({
    to: z.string(),
    subject: z.string(),
    body: z.string(),
  }),
  needsApproval: async (_context, { subject }) => {
    // check if the email is spam
    return subject.includes('spam');
  },
  execute: async ({ to, subject, body }, args) => {
    // send email
  },
});

Flow
If the agent decides to call a tool (or many) it will check if this tool needs approval by evaluating needsApproval.
If the approval is required, the agent will check if approval is already granted or rejected.
If approval has not been granted or rejected, the tool will return a static message to the agent that the tool call cannot be executed.
If approval / rejection is missing it will trigger a tool approval request.
The agent will gather all tool approval requests and interrupt the execution.
If there are any interruptions, the result will contain an interruptions array describing pending steps. A ToolApprovalItem with type: "tool_approval_item" appears when a tool call requires confirmation.
You can call result.state.approve(interruption) or result.state.reject(interruption) to approve or reject the tool call.
After handling all interruptions, you can resume execution by passing the result.state back into runner.run(agent, state) where agent is the original agent that triggered the overall run.
The flow starts again from step 1.
Example
Below is a more complete example of a human-in-the-loop flow that prompts for approval in the terminal and temporarily stores the state in a file.

Human in the loop
import { z } from 'zod';
import readline from 'node:readline/promises';
import fs from 'node:fs/promises';
import { Agent, run, tool, RunState, RunResult } from '@openai/agents';

const getWeatherTool = tool({
  name: 'get_weather',
  description: 'Get the weather for a given city',
  parameters: z.object({
    location: z.string(),
  }),
  needsApproval: async (_context, { location }) => {
    // forces approval to look up the weather in San Francisco
    return location === 'San Francisco';
  },
  execute: async ({ location }) => {
    return `The weather in ${location} is sunny`;
  },
});

const dataAgentTwo = new Agent({
  name: 'Data agent',
  instructions: 'You are a data agent',
  handoffDescription: 'You know everything about the weather',
  tools: [getWeatherTool],
});

const agent = new Agent({
  name: 'Basic test agent',
  instructions: 'You are a basic agent',
  handoffs: [dataAgentTwo],
});

async function confirm(question: string) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const answer = await rl.question(`${question} (y/n): `);
  const normalizedAnswer = answer.toLowerCase();
  rl.close();
  return normalizedAnswer === 'y' || normalizedAnswer === 'yes';
}

async function main() {
  let result: RunResult<unknown, Agent<unknown, any>> = await run(
    agent,
    'What is the weather in Oakland and San Francisco?',
  );
  let hasInterruptions = result.interruptions?.length > 0;
  while (hasInterruptions) {
    // storing
    await fs.writeFile(
      'result.json',
      JSON.stringify(result.state, null, 2),
      'utf-8',
    );

    // from here on you could run things on a different thread/process

    // reading later on
    const storedState = await fs.readFile('result.json', 'utf-8');
    const state = await RunState.fromString(agent, storedState);

    for (const interruption of result.interruptions) {
      const confirmed = await confirm(
        `Agent ${interruption.agent.name} would like to use the tool ${interruption.rawItem.name} with "${interruption.rawItem.arguments}". Do you approve?`,
      );

      if (confirmed) {
        state.approve(interruption);
      } else {
        state.reject(interruption);
      }
    }

    // resume execution of the current state
    result = await run(agent, state);
    hasInterruptions = result.interruptions?.length > 0;
  }

  console.log(result.finalOutput);
}

main().catch((error) => {
  console.dir(error, { depth: null });
});

See the full example script for a working end-to-end version.

Dealing with longer approval times
The human-in-the-loop flow is designed to be interruptible for longer periods of time without keeping your server running. If you need to shut down the request and continue later on you can serialize the state and resume later.

You can serialize the state using JSON.stringify(result.state) and resume later on by passing the serialized state into RunState.fromString(agent, serializedState) where agent is the instance of the agent that triggered the overall run.

That way you can store your serialized state in a database, or along with your request.

Versioning pending tasks
Note

This primarily applies if you are trying to store your serialized state for a longer time while doing changes to your agents.

If your approval requests take a longer time and you intend to version your agent definitions in a meaningful way or bump your Agents SDK version, we currently recommend for you to implement your own branching logic by installing two versions of the Agents SDK in parallel using package aliases.

In practice this means assigning your own code a version number and storing it along with the serialized state and guiding the deserialization to the correct version of your code.

Model Context Protocol (MCP)
The Model Context Protocol (MCP) is an open protocol that standardizes how applications provide tools and context to LLMs. From the MCP docs:

MCP is an open protocol that standardizes how applications provide context to LLMs. Think of MCP like a USB-C port for AI applications. Just as USB-C provides a standardized way to connect your devices to various peripherals and accessories, MCP provides a standardized way to connect AI models to different data sources and tools.

There are three types of MCP servers this SDK supports:

Hosted MCP server tools – remote MCP servers used as tools by the OpenAI Responses API
Streamable HTTP MCP servers – local or remote servers that implement the Streamable HTTP transport
Stdio MCP servers – servers accessed via standard input/output (the simplest option)
Choose a server type based on your use‑case:

What you need	Recommended option
Call publicly accessible remote servers with default OpenAI responses models	1. Hosted MCP tools
Use publicly accessible remote servers but have the tool calls triggered locally	2. Streamable HTTP
Use locally running Streamable HTTP servers	2. Streamable HTTP
Use any Streamable HTTP servers with non-OpenAI-Responses models	2. Streamable HTTP
Work with local MCP servers that only support the standard-I/O protocol	3. Stdio
1. Hosted MCP server tools
Hosted tools push the entire round‑trip into the model. Instead of your code calling an MCP server, the OpenAI Responses API invokes the remote tool endpoint and streams the result back to the model.

Here is the simplest example of using hosted MCP tools. You can pass the remote MCP server’s label and URL to the hostedMcpTool utility function, which is helpful for creating hosted MCP server tools.

hostedAgent.ts
import { Agent, hostedMcpTool } from '@openai/agents';

export const agent = new Agent({
  name: 'MCP Assistant',
  instructions: 'You must always use the MCP tools to answer questions.',
  tools: [
    hostedMcpTool({
      serverLabel: 'gitmcp',
      serverUrl: 'https://gitmcp.io/openai/codex',
    }),
  ],
});

Then, you can run the Agent with the run function (or your own customized Runner instance’s run method):

Run with hosted MCP tools
import { run } from '@openai/agents';
import { agent } from './hostedAgent';

async function main() {
  const result = await run(
    agent,
    'Which language is the repo I pointed in the MCP tool settings written in?',
  );
  console.log(result.finalOutput);
}

main().catch(console.error);

To stream incremental MCP results, pass stream: true when you run the Agent:

Run with hosted MCP tools (streaming)
import { run } from '@openai/agents';
import { agent } from './hostedAgent';

async function main() {
  const result = await run(
    agent,
    'Which language is the repo I pointed in the MCP tool settings written in?',
    { stream: true },
  );

  for await (const event of result) {
    if (
      event.type === 'raw_model_stream_event' &&
      event.data.type === 'model' &&
      event.data.event.type !== 'response.mcp_call_arguments.delta' &&
      event.data.event.type !== 'response.output_text.delta'
    ) {
      console.log(`Got event of type ${JSON.stringify(event.data)}`);
    }
  }
  console.log(`Done streaming; final result: ${result.finalOutput}`);
}

main().catch(console.error);

Optional approval flow
For sensitive operations you can require human approval of individual tool calls. Pass either requireApproval: 'always' or a fine‑grained object mapping tool names to 'never'/'always'.

If you can programatically determine whether a tool call is safe, you can use the onApproval callback to approve or reject the tool call. If you require human approval, you can use the same human-in-the-loop (HITL) approach using interruptions as for local function tools.

Human in the loop with hosted MCP tools
import { Agent, run, hostedMcpTool, RunToolApprovalItem } from '@openai/agents';

async function main(): Promise<void> {
  const agent = new Agent({
    name: 'MCP Assistant',
    instructions: 'You must always use the MCP tools to answer questions.',
    tools: [
      hostedMcpTool({
        serverLabel: 'gitmcp',
        serverUrl: 'https://gitmcp.io/openai/codex',
        // 'always' | 'never' | { never, always }
        requireApproval: {
          never: {
            toolNames: ['search_codex_code', 'fetch_codex_documentation'],
          },
          always: {
            toolNames: ['fetch_generic_url_content'],
          },
        },
      }),
    ],
  });

  let result = await run(agent, 'Which language is this repo written in?');
  while (result.interruptions && result.interruptions.length) {
    for (const interruption of result.interruptions) {
      // Human in the loop here
      const approval = await confirm(interruption);
      if (approval) {
        result.state.approve(interruption);
      } else {
        result.state.reject(interruption);
      }
    }
    result = await run(agent, result.state);
  }
  console.log(result.finalOutput);
}

import { stdin, stdout } from 'node:process';
import * as readline from 'node:readline/promises';

async function confirm(item: RunToolApprovalItem): Promise<boolean> {
  const rl = readline.createInterface({ input: stdin, output: stdout });
  const name = item.rawItem.name;
  const params = item.rawItem.providerData?.arguments;
  const answer = await rl.question(
    `Approve running tool (mcp: ${name}, params: ${params})? (y/n) `,
  );
  rl.close();
  return answer.toLowerCase().trim() === 'y';
}

main().catch(console.error);

Fully working samples (Hosted tools/Streamable HTTP/stdio + Streaming, HITL, onApproval) are examples/mcp in our GitHub repository.

2. Streamable HTTP MCP servers
When your Agent talks directly to a Streamable HTTP MCP server—local or remote—instantiate MCPServerStreamableHttp with the server url, name, and any optional settings:

Run with Streamable HTTP MCP servers
import { Agent, run, MCPServerStreamableHttp } from '@openai/agents';

async function main() {
  const mcpServer = new MCPServerStreamableHttp({
    url: 'https://gitmcp.io/openai/codex',
    name: 'GitMCP Documentation Server',
  });
  const agent = new Agent({
    name: 'GitMCP Assistant',
    instructions: 'Use the tools to respond to user requests.',
    mcpServers: [mcpServer],
  });

  try {
    await mcpServer.connect();
    const result = await run(agent, 'Which language is this repo written in?');
    console.log(result.finalOutput);
  } finally {
    await mcpServer.close();
  }
}

main().catch(console.error);

The constructor also accepts additional MCP TypeScript‑SDK options such as authProvider, requestInit, reconnectionOptions, and sessionId. See the MCP TypeScript SDK repository and its documents for details.

3. Stdio MCP servers
For servers that expose only standard I/O, instantiate MCPServerStdio with a fullCommand:

Run with Stdio MCP servers
import { Agent, run, MCPServerStdio } from '@openai/agents';
import * as path from 'node:path';

async function main() {
  const samplesDir = path.join(__dirname, 'sample_files');
  const mcpServer = new MCPServerStdio({
    name: 'Filesystem MCP Server, via npx',
    fullCommand: `npx -y @modelcontextprotocol/server-filesystem ${samplesDir}`,
  });
  await mcpServer.connect();
  try {
    const agent = new Agent({
      name: 'FS MCP Assistant',
      instructions:
        'Use the tools to read the filesystem and answer questions based on those files. If you are unable to find any files, you can say so instead of assuming they exist.',
      mcpServers: [mcpServer],
    });
    const result = await run(agent, 'Read the files and list them.');
    console.log(result.finalOutput);
  } finally {
    await mcpServer.close();
  }
}

main().catch(console.error);

Other things to know
For Streamable HTTP and Stdio servers, each time an Agent runs it may call list_tools() to discover available tools. Because that round‑trip can add latency—especially to remote servers—you can cache the results in memory by passing cacheToolsList: true to MCPServerStdio or MCPServerStreamableHttp.

Only enable this if you’re confident the tool list won’t change. To invalidate the cache later, call invalidateToolsCache() on the server instance.

Tracing
The Agents SDK includes built-in tracing, collecting a comprehensive record of events during an agent run: LLM generations, tool calls, handoffs, guardrails, and even custom events that occur. Using the Traces dashboard, you can debug, visualize, and monitor your workflows during development and in production.

Note

Tracing is enabled by default. There are two ways to disable tracing:

You can globally disable tracing by setting the env var OPENAI_AGENTS_DISABLE_TRACING=1
You can disable tracing for a single run by setting RunConfig.tracingDisabled to true
For organizations operating under a Zero Data Retention (ZDR) policy using OpenAI’s APIs, tracing is unavailable.

Export loop lifecycle
In most environments traces will automatically be exported on a regular interval. In the browser or in Cloudflare Workers, this functionality is disabled by default. Traces will still get exported if too many are queued up but they are not exported on a regular interval. Instead you should use getGlobalTraceProvider().forceFlush() to manually export the traces as part of your code’s lifecycle.

For example, in a Cloudflare Worker, you should wrap your code into a try/catch/finally block and use force flush with waitUntil to ensure that traces are exported before the worker exits.

import { getGlobalTraceProvider } from '@openai/agents';

export default {
  async fetch(request, env, ctx): Promise<Response> {
    try {
      // your agent code here
      return new Response(`success`);
    } catch (error) {
      console.error(error);
      return new Response(String(error), { status: 500 });
    } finally {
      // make sure to flush any remaining traces before exiting
      ctx.waitUntil(getGlobalTraceProvider().forceFlush());
    }
  },
};

Traces and spans
Traces represent a single end-to-end operation of a “workflow”. They’re composed of Spans. Traces have the following properties:
workflow_name: This is the logical workflow or app. For example “Code generation” or “Customer service”.
trace_id: A unique ID for the trace. Automatically generated if you don’t pass one. Must have the format trace_<32_alphanumeric>.
group_id: Optional group ID, to link multiple traces from the same conversation. For example, you might use a chat thread ID.
disabled: If True, the trace will not be recorded.
metadata: Optional metadata for the trace.
Spans represent operations that have a start and end time. Spans have:
started_at and ended_at timestamps.
trace_id, to represent the trace they belong to
parent_id, which points to the parent Span of this Span (if any)
span_data, which is information about the Span. For example, AgentSpanData contains information about the Agent, GenerationSpanData contains information about the LLM generation, etc.
Default tracing
By default, the SDK traces the following:

The entire run() or Runner.run() is wrapped in a Trace.
Each time an agent runs, it is wrapped in AgentSpan
LLM generations are wrapped in GenerationSpan
Function tool calls are each wrapped in FunctionSpan
Guardrails are wrapped in GuardrailSpan
Handoffs are wrapped in HandoffSpan
By default, the trace is named “Agent workflow”. You can set this name if you use withTrace, or you can can configure the name and other properties with the RunConfig.workflowName.

In addition, you can set up custom trace processors to push traces to other destinations (as a replacement, or secondary destination).

Voice agent tracing
If you are using RealtimeAgent and RealtimeSession with the default OpenAI Realtime API, tracing will automatically happen on the Realtime API side unless you disable it on the RealtimeSession using tracingDisabled: true or using the OPENAI_AGENTS_DISABLE_TRACING environment variable.

Check out the Voice agents guide for more details.

Higher level traces
Sometimes, you might want multiple calls to run() to be part of a single trace. You can do this by wrapping the entire code in a withTrace().

import { Agent, run, withTrace } from '@openai/agents';

const agent = new Agent({
  name: 'Joke generator',
  instructions: 'Tell funny jokes.',
});

await withTrace('Joke workflow', async () => {
  const result = await run(agent, 'Tell me a joke');
  const secondResult = await run(
    agent,
    `Rate this joke: ${result.finalOutput}`,
  );
  console.log(`Joke: ${result.finalOutput}`);
  console.log(`Rating: ${secondResult.finalOutput}`);
});

Because the two calls to run are wrapped in a withTrace(), the individual runs will be part of the overall trace rather than creating two traces.
Creating traces
You can use the withTrace() function to create a trace. Alternatively, you can use getGlobalTraceProvider().createTrace() to create a new trace manually and pass it into withTrace().

The current trace is tracked via a Node.js AsyncLocalStorage or the respective environment polyfills. This means that it works with concurrency automatically.

Creating spans
You can use the various create*Span() (e.g. createGenerationSpan(), createFunctionSpan(), etc.) methods to create a span. In general, you don’t need to manually create spans. A createCustomSpan() function is available for tracking custom span information.

Spans are automatically part of the current trace, and are nested under the nearest current span, which is tracked via a Node.js AsyncLocalStorage or the respective environment polyfills.

Sensitive data
Certain spans may capture potentially sensitive data.

The createGenerationSpan() stores the inputs/outputs of the LLM generation, and createFunctionSpan() stores the inputs/outputs of function calls. These may contain sensitive data, so you can disable capturing that data via RunConfig.traceIncludeSensitiveData .

Custom tracing processors
The high level architecture for tracing is:

At initialization, we create a global TraceProvider, which is responsible for creating traces and can be accessed through getGlobalTraceProvider().
We configure the TraceProvider with a BatchTraceProcessor that sends traces/spans in batches to a OpenAITracingExporter, which exports the spans and traces to the OpenAI backend in batches.
To customize this default setup, to send traces to alternative or additional backends or modifying exporter behavior, you have two options:

addTraceProcessor() lets you add an additional trace processor that will receive traces and spans as they are ready. This lets you do your own processing in addition to sending traces to OpenAI’s backend.
setTraceProcessors() lets you replace the default processors with your own trace processors. This means traces will not be sent to the OpenAI backend unless you include a TracingProcessor that does so.
External tracing processors list
AgentOps
Keywords AI

Configuring the SDK
API keys and clients
By default the SDK reads the OPENAI_API_KEY environment variable when first imported. If setting the variable is not possible you can call setDefaultOpenAIKey() manually.

Set default OpenAI key
import { setDefaultOpenAIKey } from '@openai/agents';

setDefaultOpenAIKey(process.env.OPENAI_API_KEY!); // sk-...

You may also pass your own OpenAI client instance. The SDK will otherwise create one automatically using the default key.

Set default OpenAI client
import { OpenAI } from 'openai';
import { setDefaultOpenAIClient } from '@openai/agents';

const customClient = new OpenAI({ baseURL: '...', apiKey: '...' });
setDefaultOpenAIClient(customClient);

Finally you can switch between the Responses API and the Chat Completions API.

Set OpenAI API
import { setOpenAIAPI } from '@openai/agents';

setOpenAIAPI('chat_completions');

Tracing
Tracing is enabled by default and uses the OpenAI key from the section above. A separate key may be set via setTracingExportApiKey().

Set tracing export API key
import { setTracingExportApiKey } from '@openai/agents';

setTracingExportApiKey('sk-...');

Tracing can also be disabled entirely.

Disable tracing
import { setTracingDisabled } from '@openai/agents';

setTracingDisabled(true);

Debug logging
The SDK uses the debug package for debug logging. Set the DEBUG environment variable to openai-agents* to see verbose logs.

Terminal window
export DEBUG=openai-agents*

You can obtain a namespaced logger for your own modules using getLogger(namespace) from @openai/agents.

Get logger
import { getLogger } from '@openai/agents';

const logger = getLogger('my-app');
logger.debug('something happened');

Sensitive data in logs
Certain logs may contain user data. Disable them by setting these environment variables.

To disable logging LLM inputs and outputs:

Terminal window
export OPENAI_AGENTS_DONT_LOG_MODEL_DATA=1

To disable logging tool inputs and outputs:

Terminal window
export OPENAI_AGENTS_DONT_LOG_TOOL_DATA=1

Troubleshooting
Supported environments
The OpenAI Agents SDK is supported on the following server environments:

Node.js 22+
Deno 2.35+
Bun 1.2.5+
Limited support
Cloudflare Workers: The Agents SDK can be used in Cloudflare Workers, but currently comes with some limitations:
The SDK current requires nodejs_compat to be enabled
Traces need to be manually flushed at the end of the request. See the tracing guide for more details.
Due to Cloudflare Workers’ limited support for AsyncLocalStorage some traces might not be accurate
Browsers:
Tracing is currently not supported in browsers
v8 isolates:
While you should be able to bundle the SDK for v8 isolates if you use a bundler with the right browser polyfills, tracing will not work
v8 isolates have not been extensively tested
Debug logging
If you are running into problems with the SDK, you can enable debug logging to get more information about what is happening.

Enable debug logging by setting the DEBUG environment variable to openai-agents:*.

Terminal window
DEBUG=openai-agents:*

Alternatively, you can scope the debugging to specific parts of the SDK:

openai-agents:core — for the main execution logic of the SDK
openai-agents:openai — for the OpenAI API calls
openai-agents:realtime — for the Realtime Agents components