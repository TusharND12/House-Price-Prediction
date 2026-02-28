import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts'
import type { PlaygroundInputs } from '../App'

const API = '/api'

type Props = {
  dark: boolean
  inputs: PlaygroundInputs
  onInputsChange: (inputs: PlaygroundInputs) => void
}

export default function PredictionPlayground({ dark, inputs, onInputsChange }: Props) {
  const [, setInfo] = useState<{ dataset?: string }>({})
  const [prediction, setPrediction] = useState<number | null>(null)
  const [contributions, setContributions] = useState<{ name: string; contribution: number; percent: number }[]>([])
  const [loading, setLoading] = useState(false)
  const [budget, setBudget] = useState<string>('')
  const setInputs = (next: PlaygroundInputs | ((prev: PlaygroundInputs) => PlaygroundInputs)) => {
    onInputsChange(typeof next === 'function' ? next(inputs) : next)
  }

  useEffect(() => {
    fetch(`${API}/info`).then(r => r.json()).then(setInfo)
  }, [])

  const predict = () => {
    setLoading(true)
    fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputs),
    })
      .then(r => r.json())
      .then((res) => {
        if (res.error) return
        setPrediction(res.prediction)
        setContributions(res.contributions || [])
      })
      .finally(() => setLoading(false))
  }

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4">Prediction Playground</h2>
        <p className="text-slate-600 dark:text-slate-400 mb-6">
          Adjust the sliders to design your ideal property. Get an instant price prediction.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow border border-slate-200 dark:border-slate-700">
            <h3 className="font-semibold mb-4 text-slate-800 dark:text-white">Property Details</h3>
            <div className="space-y-4">
              {[
                { key: 'lot_area', label: 'Lot Area (sq ft)', min: 1000, max: 50000, step: 100 },
                { key: 'overall_qual', label: 'Overall Quality (1-10)', min: 1, max: 10, step: 1 },
                { key: 'gr_liv_area', label: 'Above Ground Living Area', min: 300, max: 6000, step: 50 },
                { key: 'garage_cars', label: 'Garage Cars', min: 0, max: 4, step: 1 },
                { key: 'total_bsmt_sf', label: 'Total Basement (sq ft)', min: 0, max: 6000, step: 50 },
                { key: 'year_built', label: 'Year Built', min: 1880, max: 2010, step: 1 },
                { key: 'full_bath', label: 'Full Bathrooms', min: 0, max: 4, step: 1 },
                { key: 'fireplace', label: 'Fireplaces', min: 0, max: 4, step: 1 },
              ].map(({ key, label, min, max, step }) => (
                <div key={key}>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">{label}</label>
                  <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={inputs[key as keyof typeof inputs]}
                    onChange={(e) => setInputs({ ...inputs, [key]: Number(e.target.value) })}
                    className="w-full"
                  />
                  <span className="text-sm text-slate-500">{inputs[key as keyof typeof inputs]}</span>
                </div>
              ))}
            </div>
            <button
              onClick={predict}
              disabled={loading}
              className="mt-6 w-full py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg disabled:opacity-50"
            >
              {loading ? 'Predicting...' : 'Predict Price'}
            </button>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow border border-slate-200 dark:border-slate-700">
            <h3 className="font-semibold mb-4 text-slate-800 dark:text-white">Prediction</h3>
            {prediction !== null ? (
              <>
                <div className="text-4xl font-bold text-indigo-600 dark:text-indigo-400 mb-4">
                  ${prediction.toLocaleString()}
                </div>
                <div className="mb-6 p-4 bg-slate-100 dark:bg-slate-700 rounded-lg">
                  <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">Budget Mode: Enter target ($)</label>
                  <input
                    type="number"
                    placeholder="e.g. 200000"
                    value={budget}
                    onChange={(e) => setBudget(e.target.value)}
                    className="w-full px-3 py-2 rounded border dark:bg-slate-800 dark:border-slate-600"
                  />
                  {budget && Number(budget) > 0 && (
                    <p className={`mt-2 text-sm font-medium ${prediction <= Number(budget) ? 'text-green-600' : 'text-red-600'}`}>
                      {prediction <= Number(budget)
                        ? `Under budget by $${(Number(budget) - prediction).toLocaleString()}`
                        : `Over budget by $${(prediction - Number(budget)).toLocaleString()}`}
                    </p>
                  )}
                </div>
                <h4 className="font-medium mb-3 text-slate-700 dark:text-slate-300">Feature Contributions</h4>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={contributions.slice(0, 10)} layout="vertical" margin={{ left: 80 }}>
                      <XAxis type="number" stroke={dark ? '#94a3b8' : '#64748b'} />
                      <YAxis type="category" dataKey="name" width={75} stroke={dark ? '#94a3b8' : '#64748b'} />
                      <Tooltip contentStyle={{ background: dark ? '#1e293b' : '#fff', border: '1px solid #334155' }} />
                      <Bar dataKey="contribution" fill="#6366f1" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </>
            ) : (
              <p className="text-slate-500">Click Predict Price to see the result.</p>
            )}
          </div>
        </div>
      </section>
    </div>
  )
}
