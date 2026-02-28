import { useState, useEffect } from 'react'
import { Shield } from 'lucide-react'
import type { PlaygroundInputs } from '../App'

const API = '/api'

type Props = { dark: boolean; inputs: PlaygroundInputs }

export default function FairnessLens({ dark, inputs: _inputs }: Props) {
  const [data, setData] = useState<{ message: string; caveats: string[]; recommendation: string } | null>(null)

  useEffect(() => {
    fetch(`${API}/fairness`)
      .then(r => r.json())
      .then(setData)
  }, [])

  if (!data) return null

  return (
    <div className="space-y-6">
      <h3 className="font-semibold text-slate-800 dark:text-white flex items-center gap-2">
        <Shield size={20} /> Fairness Lens
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400">
        Responsible AI: what this model does and doesn&apos;t consider.
      </p>
      <div className="p-4 rounded-lg bg-slate-100 dark:bg-slate-700">
        <p className="text-slate-700 dark:text-slate-200 mb-4">{data.message}</p>
        <ul className="list-disc list-inside space-y-1 text-sm text-slate-600 dark:text-slate-400 mb-4">
          {data.caveats?.map((c, i) => <li key={i}>{c}</li>)}
        </ul>
        <p className="text-sm font-medium text-indigo-600 dark:text-indigo-400">{data.recommendation}</p>
      </div>
    </div>
  )
}
