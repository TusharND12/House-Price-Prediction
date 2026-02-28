import { useState } from 'react'
import { Home, BarChart3, GitCompare, Map, Moon, Sun, Sparkles } from 'lucide-react'
import PredictionPlayground from './components/PredictionPlayground'
import ExplainabilityHub from './components/ExplainabilityHub'
import WhatIfLab from './components/WhatIfLab'
import MapView from './components/MapView'
import ExperimentsHub from './components/ExperimentsHub'

export const DEFAULT_PLAYGROUND_INPUTS = {
  lot_area: 8450,
  overall_qual: 7,
  gr_liv_area: 1710,
  garage_cars: 2,
  total_bsmt_sf: 856,
  year_built: 2003,
  full_bath: 2,
  fireplace: 0,
}

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
      <header className="border-b border-slate-200 dark:border-slate-700 dark:bg-slate-800 bg-white sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold text-slate-800 dark:text-white">
            SmartExplain AI
          </h1>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setDark(!dark)}
              className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700"
            >
              {dark ? <Sun size={20} /> : <Moon size={20} />}
            </button>
            <nav className="flex gap-1">
              {tabs.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition ${
                    tab === t.id
                      ? 'bg-indigo-600 text-white'
                      : 'text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
                  }`}
                >
                  {t.icon}
                  {t.label}
                </button>
              ))}
            </nav>
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
