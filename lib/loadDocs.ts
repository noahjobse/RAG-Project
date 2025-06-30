import { OpenAIEmbeddings, ChatOpenAI } from '@langchain/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { Document } from 'langchain/document';
import fs from 'fs';
import path from 'path';

const openAIApiKey = process.env.OPENAI_API_KEY;
if (!openAIApiKey) throw new Error('Missing OPENAI_API_KEY in environment variables.');

export const embeddings = new OpenAIEmbeddings({
  modelName: 'text-embedding-3-large',
  openAIApiKey,
});

export const llm = new ChatOpenAI({
  modelName: 'gpt-4o',
  temperature: 0.2,
  openAIApiKey,
});

const docsPath = path.join(process.cwd(), 'documents');

export async function loadLocalDocuments(): Promise<Document[]> {
  const files = fs.readdirSync(docsPath);
  const allDocs: Document[] = [];

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  for (const file of files) {
    const ext = path.extname(file);
    if (!['.txt', '.md'].includes(ext)) continue;

    const fullPath = path.join(docsPath, file);
    const content = fs.readFileSync(fullPath, 'utf-8');

    const splitDocs = await splitter.createDocuments(
      [content],
      [{ source: file }] // metadata
    );

    allDocs.push(...splitDocs);
  }

  return allDocs;
}
