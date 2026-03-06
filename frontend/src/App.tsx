import { useState } from 'react'
import { Home, BarChart3, GitCompare, Map, Moon, Sun, Sparkles } from 'lucide-react'
import PredictionPlayground from './components/PredictionPlayground'
import ExplainabilityHub from './components/ExplainabilityHub'
import WhatIfLab from './components/WhatIfLab'
import MapView from './components/MapView'
import ExperimentsHub from './components/ExperimentsHub'

export const DEFAULT_PLAYGROUND_INPUTS_HP = {
  total_rooms: 2635,
  total_bedrooms: 537,
  housing_median_age: 29,
  median_income: 3.87,
  population: 1425,
  households: 499,
  longitude: -119,
  latitude: 36,
  ocean_proximity: 'INLAND',
}

export const DEFAULT_PLAYGROUND_INPUTS = DEFAULT_PLAYGROUND_INPUTS_HP

export type PlaygroundInputs = typeof DEFAULT_PLAYGROUND_INPUTS

type Tab = 'playground' | 'explain' | 'whatif' | 'map' | 'experiments'

export default function App() {
  const [tab, setTab] = useState<Tab>('playground')
  const [dark, setDark] = useState(false)
  const [playgroundInputs, setPlaygroundInputs] = useState(DEFAULT_PLAYGROUND_INPUTS)

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'playground', label: 'Prediction Playground', icon: <Home size={18} /> },
    { id: 'explain', label: 'Explainability Hub', icon: <BarChart3 size={18} /> },
    { id: 'whatif', label: 'What-If Lab', icon: <GitCompare size={18} /> },
    { id: 'map', label: 'Map Explorer', icon: <Map size={18} /> },
    { id: 'experiments', label: 'Never Seen Before', icon: <Sparkles size={18} /> },
  ]

  return (
    <div className={dark ? 'dark bg-slate-900 min-h-screen' : 'bg-slate-50 min-h-screen'}>
      <header className="border-b border-slate-200 dark:border-slate-800 bg-white/95 dark:bg-slate-900/95 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3 md:py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-lg bg-indigo-600 flex items-center justify-center text-white text-sm font-semibold shadow-sm">
              SE
            </div>
            <div>
              <h1 className="text-lg md:text-xl font-semibold tracking-tight text-slate-900 dark:text-white">
                SmartExplain AI
              </h1>
              <p className="text-[0.7rem] text-slate-500 dark:text-slate-400">
                House price explainability playground
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <nav className="hidden sm:flex items-center gap-1.5">
              {tabs.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition ${
                    tab === t.id
                      ? 'bg-indigo-600 text-white shadow-sm'
                      : 'text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800'
                  }`}
                >
                  {t.icon}
                  {t.label}
                </button>
              ))}
            </nav>
            <button
              onClick={() => setDark(!dark)}
              className="p-2 rounded-full border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              {dark ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {tab === 'playground' && (
          <PredictionPlayground
            dark={dark}
            inputs={playgroundInputs}
            onInputsChange={setPlaygroundInputs}
          />
        )}
        {tab === 'explain' && <ExplainabilityHub dark={dark} />}
        {tab === 'whatif' && <WhatIfLab dark={dark} />}
        {tab === 'map' && <MapView />}
        {tab === 'experiments' && (
          <ExperimentsHub dark={dark} inputs={playgroundInputs} />
        )}
      </main>
    </div>
  )
}
