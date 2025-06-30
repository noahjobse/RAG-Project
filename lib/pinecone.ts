import { PineconeClient } from '@pinecone-database/pinecone';

const pinecone = new PineconeClient();
let initialized = false;

export async function initPinecone() {
  if (!initialized) {
    await pinecone.init({
      apiKey: process.env.PINECONE_API_KEY!,
      environment: process.env.PINECONE_ENVIRONMENT!,
    });
    initialized = true;
  }

  return pinecone.Index(process.env.PINECONE_INDEX_NAME!);
}
