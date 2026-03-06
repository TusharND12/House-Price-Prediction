import { useState } from 'react'
import { GitCompare } from 'lucide-react'

const API = '/api'

const defaultInputs = {
  total_rooms: 2635,
  total_bedrooms: 537,
  housing_median_age: 29,
  median_income: 3.87,
  population: 1425,
  households: 499,
  longitude: -119,
  latitude: 36,
  ocean_proximity: 'INLAND' as const,
}

export default function WhatIfLab() {
  const [current, setCurrent] = useState(defaultInputs)
  const [updated, setUpdated] = useState({ ...defaultInputs, total_rooms: 4000 })
  const [result, setResult] = useState<{
    original_prediction?: number
    updated_prediction?: number
    price_difference?: number
  } | { error?: string } | null>(null)
  const [loading, setLoading] = useState(false)
  const [resultKey, setResultKey] = useState(0)

  const compare = () => {
    setLoading(true)
    fetch(`${API}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ current, updated }),
    })
      .then(r => r.json())
      .then((res) => {
        setResult(res)
        if (!('error' in res)) {
          setResultKey((k) => k + 1)
        }
      })
      .finally(() => setLoading(false))
  }

  const fields = [
    { key: 'total_rooms', label: 'Total Rooms' },
    { key: 'total_bedrooms', label: 'Total Bedrooms' },
    { key: 'housing_median_age', label: 'Housing Age' },
    { key: 'median_income', label: 'Median Income' },
    { key: 'population', label: 'Population' },
    { key: 'households', label: 'Households' },
    { key: 'longitude', label: 'Longitude' },
    { key: 'latitude', label: 'Latitude' },
    { key: 'ocean_proximity', label: 'Ocean Proximity', isSelect: true },
  ]

  return (
    <div className="space-y-8">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-white">What-If Lab</h2>
      <p className="text-slate-600 dark:text-slate-400">
        Compare two property scenarios side by side. See how changes affect the predicted price.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow border border-slate-200 dark:border-slate-700">
          <h3 className="font-semibold mb-4 text-slate-800 dark:text-white">Scenario A (Current)</h3>
          {fields.map(({ key, label, isSelect }) => (
            <div key={key} className="mb-3">
              <label className="block text-sm text-slate-600 dark:text-slate-400">{label}</label>
              {isSelect ? (
                <select
                  value={current[key as keyof typeof current] as string}
                  onChange={(e) => setCurrent({ ...current, [key]: e.target.value })}
                  className="w-full px-3 py-2 rounded border border-slate-300 dark:border-slate-600 dark:bg-slate-700 dark:text-white"
                >
                  {['INLAND', '<1H OCEAN', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'].map((v) => (
                    <option key={v} value={v}>{v}</option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  value={current[key as keyof typeof current] as number}
                  onChange={(e) => setCurrent({ ...current, [key]: Number(e.target.value) })}
                  className="w-full px-3 py-2 rounded border border-slate-300 dark:border-slate-600 dark:bg-slate-700 dark:text-white"
                />
              )}
            </div>
          ))}
        </div>

        <div className="flex flex-col items-center justify-center">
          <button
            onClick={compare}
            disabled={loading}
            className={`p-4 bg-indigo-600 hover:bg-indigo-700 text-white rounded-full shadow transition-transform ${
              loading ? 'animate-spin-slow' : ''
            } disabled:opacity-60 disabled:cursor-not-allowed`}
          >
            <GitCompare size={24} />
          </button>
          <p className="mt-2 text-sm text-slate-500">
            {loading ? 'Comparing scenarios…' : 'Compare'}
          </p>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow border border-slate-200 dark:border-slate-700">
          <h3 className="font-semibold mb-4 text-slate-800 dark:text-white">Scenario B (What-If)</h3>
          {fields.map(({ key, label, isSelect }) => (
            <div key={key} className="mb-3">
              <label className="block text-sm text-slate-600 dark:text-slate-400">{label}</label>
              {isSelect ? (
                <select
                  value={updated[key as keyof typeof updated] as string}
                  onChange={(e) => setUpdated({ ...updated, [key]: e.target.value })}
                  className="w-full px-3 py-2 rounded border border-slate-300 dark:border-slate-600 dark:bg-slate-700 dark:text-white"
                >
                  {['INLAND', '<1H OCEAN', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'].map((v) => (
                    <option key={v} value={v}>{v}</option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  value={updated[key as keyof typeof updated] as number}
                  onChange={(e) => setUpdated({ ...updated, [key]: Number(e.target.value) })}
                  className="w-full px-3 py-2 rounded border border-slate-300 dark:border-slate-600 dark:bg-slate-700 dark:text-white"
                />
              )}
            </div>
          ))}
        </div>
      </div>

      {result && !('error' in result) && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-slate-100 dark:bg-slate-800 rounded-xl p-6 text-center">
              <p className="text-sm text-slate-600 dark:text-slate-400">Scenario A</p>
              <p className="text-2xl font-bold text-slate-800 dark:text-white">
                ${(result.original_prediction || 0).toLocaleString()}
              </p>
            </div>
            <div className="bg-indigo-50 dark:bg-indigo-900/30 rounded-xl p-6 text-center">
              <p className="text-sm text-slate-600 dark:text-slate-400">Price Difference</p>
              <p className={`text-2xl font-bold ${(result.price_difference || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {(result.price_difference || 0) >= 0 ? '+' : ''}${(result.price_difference || 0).toLocaleString()}
              </p>
            </div>
            <div className="bg-slate-100 dark:bg-slate-800 rounded-xl p-6 text-center">
              <p className="text-sm text-slate-600 dark:text-slate-400">Scenario B</p>
              <p className="text-2xl font-bold text-slate-800 dark:text-white">
                ${(result.updated_prediction || 0).toLocaleString()}
              </p>
            </div>
          </div>

          <div
            key={resultKey}
            className="mt-4 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700 p-5 space-y-4 shadow-sm"
          >
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                  Animated comparison
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Watch how the price slides from Scenario A to Scenario B.
                </p>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {result.original_prediction && result.updated_prediction
                  ? `${(result.original_prediction).toLocaleString()} → ${(result.updated_prediction).toLocaleString()}`
                  : ''}
              </p>
            </div>
            <WhatIfAnimation
              a={result.original_prediction || 0}
              b={result.updated_prediction || 0}
            />
          </div>
        </>
      )}
    </div>
  )
}

type AnimProps = { a: number; b: number }

function WhatIfAnimation({ a, b }: AnimProps) {
  const max = Math.max(a, b, 1)
  const aPct = (a / max) * 100
  const bPct = (b / max) * 100
  const diff = b - a
  const label =
    diff === 0
      ? 'No change in price'
      : diff > 0
      ? 'Scenario B is more expensive'
      : 'Scenario B is cheaper'

  return (
    <div className="space-y-2 text-xs">
      <div className="h-10 rounded-full bg-slate-100 dark:bg-slate-800 relative overflow-hidden">
        <div
          className="se-compare-bar se-compare-bar-a"
          style={{ width: `${aPct}%` }}
        />
        <div
          className="se-compare-bar se-compare-bar-b"
          style={{ width: `${bPct}%` }}
        />
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <span className="px-2 py-0.5 rounded-full bg-white/80 dark:bg-slate-900/80 text-[0.7rem] text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-700">
            {label}
          </span>
        </div>
      </div>
      <div className="flex justify-between text-[0.7rem] text-slate-500 dark:text-slate-400">
        <span>Scenario A</span>
        <span>Scenario B</span>
      </div>
    </div>
  )
}
