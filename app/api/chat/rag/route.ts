import { NextRequest, NextResponse } from 'next/server';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { PineconeStore } from '@langchain/community/vectorstores/pinecone';
import { loadLocalDocuments } from '@/lib/loadDocs';
import { Pinecone } from '@pinecone-database/pinecone';
import { SystemMessage, HumanMessage } from '@langchain/core/messages';

// 🔐 ENV validation
const openAIApiKey = process.env.OPENAI_API_KEY!;
const pineconeApiKey = process.env.PINECONE_API_KEY!;
const pineconeEnv = process.env.PINECONE_ENVIRONMENT!;
const pineconeIndexName = process.env.PINECONE_INDEX_NAME!;

if (!openAIApiKey || !pineconeApiKey || !pineconeEnv || !pineconeIndexName) {
  throw new Error('Missing one or more required env vars: OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME');
}

export async function POST(req: NextRequest) {
  const { query } = await req.json();
  if (!query) {
    return NextResponse.json({ error: 'Missing query' }, { status: 400 });
  }

  // 🧠 Init LLM + Embeddings
  const llm = new ChatOpenAI({
    modelName: 'gpt-4o',
    temperature: 0.2,
    openAIApiKey,
  });

  const embeddings = new OpenAIEmbeddings({
    modelName: 'text-embedding-3-large',
    openAIApiKey,
  });

  // 📦 Init Pinecone client + index
  const pinecone = new Pinecone({
    apiKey: pineconeApiKey,
    environment: pineconeEnv,
  });

  const pineconeIndex = pinecone.index(pineconeIndexName);

  // 📄 Load local docs and embed them
  const docs = await loadLocalDocuments();
  const vectorStore = await PineconeStore.fromDocuments(docs, embeddings, {
    pineconeIndex,
  });

  // 🔍 Embed query and search
  const embeddedQuery = await embeddings.embedQuery(query);
  const results = await vectorStore.similaritySearchVectorWithScore(embeddedQuery, 4);
  const context = results.map(([doc]) => doc.pageContent).join('\n\n');

  // 💬 Invoke the model using BaseMessage objects
  const response = await llm.invoke([
    new SystemMessage('You are a helpful assistant.'),
    new HumanMessage(`Context:\n${context}\n\nQ: ${query}\nA:`),
  ]);

  return NextResponse.json({
    choices: [
      {
        message: {
          role: 'assistant',
          content: response.content,
        },
      },
    ],
  });
}
