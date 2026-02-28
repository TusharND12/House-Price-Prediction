import { useState } from 'react'
import { GitCompare } from 'lucide-react'

const API = '/api'

const defaultInputs = {
  lot_area: 8450,
  overall_qual: 7,
  gr_liv_area: 1710,
  garage_cars: 2,
  total_bsmt_sf: 856,
  year_built: 2003,
  full_bath: 2,
  fireplace: 0,
}

export default function WhatIfLab() {
  const [current, setCurrent] = useState(defaultInputs)
  const [updated, setUpdated] = useState({ ...defaultInputs, gr_liv_area: 2500 })
  const [result, setResult] = useState<{
    original_prediction?: number
    updated_prediction?: number
    price_difference?: number
  } | { error?: string } | null>(null)

  const compare = () => {
    fetch(`${API}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ current, updated }),
    })
      .then(r => r.json())
      .then(setResult)
  }

  const fields = [
    { key: 'lot_area', label: 'Lot Area' },
    { key: 'overall_qual', label: 'Overall Qual' },
    { key: 'gr_liv_area', label: 'Living Area' },
    { key: 'garage_cars', label: 'Garage Cars' },
    { key: 'total_bsmt_sf', label: 'Basement' },
    { key: 'year_built', label: 'Year Built' },
    { key: 'full_bath', label: 'Bathrooms' },
    { key: 'fireplace', label: 'Fireplaces' },
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
          {fields.map(({ key, label }) => (
            <div key={key} className="mb-3">
              <label className="block text-sm text-slate-600 dark:text-slate-400">{label}</label>
              <input
                type="number"
                value={current[key as keyof typeof current]}
                onChange={(e) => setCurrent({ ...current, [key]: Number(e.target.value) })}
                className="w-full px-3 py-2 rounded border border-slate-300 dark:border-slate-600 dark:bg-slate-700 dark:text-white"
              />
            </div>
          ))}
        </div>

        <div className="flex flex-col items-center justify-center">
          <button
            onClick={compare}
            className="p-4 bg-indigo-600 hover:bg-indigo-700 text-white rounded-full"
          >
            <GitCompare size={24} />
          </button>
          <p className="mt-2 text-sm text-slate-500">Compare</p>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow border border-slate-200 dark:border-slate-700">
          <h3 className="font-semibold mb-4 text-slate-800 dark:text-white">Scenario B (What-If)</h3>
          {fields.map(({ key, label }) => (
            <div key={key} className="mb-3">
              <label className="block text-sm text-slate-600 dark:text-slate-400">{label}</label>
              <input
                type="number"
                value={updated[key as keyof typeof updated]}
                onChange={(e) => setUpdated({ ...updated, [key]: Number(e.target.value) })}
                className="w-full px-3 py-2 rounded border border-slate-300 dark:border-slate-600 dark:bg-slate-700 dark:text-white"
              />
            </div>
          ))}
        </div>
      </div>

      {result && !('error' in result) && (
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
      )}
    </div>
  )
}
