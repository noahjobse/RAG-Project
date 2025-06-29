// app/api/rag/route.ts
import { NextRequest, NextResponse } from 'next/server'
import { embedQuery } from '@/lib/embeddings'
import { getRelevantDocs } from '@/lib/vectorStore'
import { getLLMResponse } from '@/lib/generate'

export async function POST(req: NextRequest) {
  const { query } = await req.json()

  const embedded = await embedQuery(query)
  const docs = await getRelevantDocs(embedded)

  const context = docs.map(d => d.pageContent).join('\n')
  const prompt = `Context:\n${context}\n\nQ: ${query}\nA:`

  const answer = await getLLMResponse(prompt)

  return NextResponse.json({
    choices: [
      {
        message: {
          role: 'assistant',
          content: answer,
        },
      },
    ],
  })
}
