import { useCallback, useEffect, useMemo, useState } from 'react'
import { BriefcaseMedical } from 'lucide-react'
import './App.css'

const API_BASE = import.meta.env.DEV ? '/api' : 'http://127.0.0.1:8000'

const LogoIcon = () => (
  <BriefcaseMedical size={32} className="logo-icon" aria-hidden />
)

function toAbsoluteImageUrl(path) {
  if (!path) return null
  if (path.startsWith('http://') || path.startsWith('https://')) return path
  return `${API_BASE}${path}`
}

function buildDefaultLabels(candidates) {
  const out = {}
  for (const c of candidates ?? []) out[c.vector_id] = false
  return out
}

function ReplayBufferPage() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [data, setData] = useState(null)
  const [limit, setLimit] = useState(50)

  const load = useCallback(async () => {
    setError(null)
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/online/replay_buffer?limit=${Number(limit)}`)
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText || 'Failed to load replay buffer')
      }
      setData(await res.json())
    } catch (e) {
      setError(e.message || 'Failed to load replay buffer')
    } finally {
      setLoading(false)
    }
  }, [limit])

  useEffect(() => {
    load()
    const id = setInterval(load, 4000)
    return () => clearInterval(id)
  }, [load])

  return (
    <div className="app">
      <div className="content">
        <header className="header">
          <div className="brand">
            <LogoIcon />
            <div>
              <h1 className="brand-title">VeriMed</h1>
              <p className="brand-subtitle">Replay Buffer Monitor</p>
            </div>
          </div>
        </header>

        <div className="output-section">
          <div className="actions actions-row">
            <label className="setting-item">
              <span>Limit</span>
              <input type="number" min={1} max={1000} value={limit} onChange={(e) => setLimit(e.target.value)} />
            </label>
            <button type="button" className="scan-btn scan-btn--active" onClick={load} disabled={loading}>
              {loading ? 'Refreshing…' : 'Refresh'}
            </button>
          </div>
          {error && <div className="error">{error}</div>}
          {data && (
            <>
              <p className="find-meta">
                Replay size: {data.replay?.size} (alpha={data.replay?.alpha}, beta={data.replay?.beta}) | online_feedback: {data.online_feedback?.count} | lookalike_graph: {data.lookalike_graph?.count}
              </p>
              <div className="result-columns">
                <div className="result-list lookalikes">
                  <h3>Replay (latest)</h3>
                  <ul>
                    {(data.replay?.items ?? []).map((x) => (
                      <li key={`r-${x.buffer_index}`}>
                        <span className="name">[{x.buffer_index}] y={x.label} q={x.query_vector_id} c={x.candidate_vector_id}</span>
                        <span className="score">{Number(x.priority).toFixed(3)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="result-list not-lookalikes">
                  <h3>online_feedback (latest)</h3>
                  <ul>
                    {(data.online_feedback?.items ?? []).map((x) => (
                      <li key={`f-${x.id}`}>
                        <span className="name">#{x.id} y={x.label} q={x.query_vector_id} c={x.candidate_vector_id}</span>
                        <span className="score">{x.created_at}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              <div className="output-section" style={{ marginTop: '1rem' }}>
                <h3 className="output-title">lookalike_graph (latest)</h3>
                <ul className="result-list-ul">
                  {(data.lookalike_graph?.items ?? []).map((x) => (
                    <li key={`g-${x.id}`} className="graph-row">
                      #{x.id} | base={x.base_id} ({x.base_file_name ?? '-'}) | similar={x.similar_id} ({x.similar_file_name ?? '-'})
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const pathname = typeof window !== 'undefined' ? window.location.pathname : '/'
  if (pathname === '/replay_buffer') {
    return <ReplayBufferPage />
  }

  const [backendActive, setBackendActive] = useState(null)
  const [error, setError] = useState(null)
  const [loadingInit, setLoadingInit] = useState(false)
  const [loadingSubmit, setLoadingSubmit] = useState(false)
  const [batch, setBatch] = useState(null)
  const [labels, setLabels] = useState({})

  const [similarityType, setSimilarityType] = useState('l2')
  const [topK, setTopK] = useState(20)
  const [perAlpha, setPerAlpha] = useState(0.6)
  const [perBeta, setPerBeta] = useState(0.4)
  const [batchSize, setBatchSize] = useState(32)
  const [trainSteps, setTrainSteps] = useState(1)
  const [minBufferSize, setMinBufferSize] = useState(32)
  const [recentRatio, setRecentRatio] = useState(0.5)

  useEffect(() => {
    const check = () => {
      fetch(`${API_BASE}/health`)
        .then((res) => setBackendActive(res.ok))
        .catch(() => setBackendActive(false))
    }
    check()
    const id = setInterval(check, 20000)
    return () => clearInterval(id)
  }, [])

  const initSession = useCallback(async () => {
    setError(null)
    setLoadingInit(true)
    try {
      const res = await fetch(`${API_BASE}/online/session/init`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          similarity_type: similarityType,
          top_k: Number(topK),
          per_alpha: Number(perAlpha),
          per_beta: Number(perBeta),
          batch_size: Number(batchSize),
          train_steps: Number(trainSteps),
          min_buffer_size: Number(minBufferSize),
          recent_ratio: Number(recentRatio),
          hydrate_from_feedback: false,
        }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText || 'Session init failed')
      }
      const data = await res.json()
      setBatch(data.batch)
      setLabels(buildDefaultLabels(data.batch?.candidates))
    } catch (e) {
      setError(e.message || 'Session init failed')
    } finally {
      setLoadingInit(false)
    }
  }, [similarityType, topK, perAlpha, perBeta, batchSize, trainSteps, minBufferSize, recentRatio])

  const nextBatch = useCallback(async () => {
    setError(null)
    setLoadingInit(true)
    try {
      const res = await fetch(`${API_BASE}/online/session/next`)
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText || 'Next batch failed')
      }
      const data = await res.json()
      setBatch(data)
      setLabels(buildDefaultLabels(data?.candidates))
    } catch (e) {
      setError(e.message || 'Next batch failed')
    } finally {
      setLoadingInit(false)
    }
  }, [])

  const submitAndTrain = useCallback(async () => {
    if (!batch?.query?.vector_id) {
      setError('No active batch. Initialize session first.')
      return
    }
    setError(null)
    setLoadingSubmit(true)
    try {
      const payload = (batch.candidates ?? []).map((c) => ({
        vector_id: c.vector_id,
        label: labels[c.vector_id] ? 1 : 0,
      }))
      const res = await fetch(`${API_BASE}/online/session/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query_vector_id: batch.query.vector_id,
          labels: payload,
        }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText || 'Submit failed')
      }
      const data = await res.json()
      setBatch(data.next_batch)
      setLabels(buildDefaultLabels(data.next_batch?.candidates))
    } catch (e) {
      setError(e.message || 'Submit failed')
    } finally {
      setLoadingSubmit(false)
    }
  }, [batch, labels])

  const labeledCount = useMemo(
    () => Object.values(labels).filter(Boolean).length,
    [labels],
  )

  return (
    <div className="app">
      <div className="content">
        <header className="header">
          <div className="brand">
            <LogoIcon />
            <div>
              <h1 className="brand-title">VeriMed</h1>
              <p className="brand-subtitle">Medication Detection Verification System</p>
            </div>
          </div>
          <div className="system-status">
            <span
              className={`status-dot ${backendActive === true ? 'status-dot--active' : 'status-dot--inactive'}`}
              aria-hidden
            />
            <span>{backendActive === true ? 'System Active' : 'System Inactive'}</span>
          </div>
        </header>

        <div className="instructions-panel">
          <h3 className="instructions-title">Online Labeling + Training Loop</h3>
          <ol className="instructions-list">
            <li>Set retrieval mode and top-k, then click <strong>Initialize Session</strong>.</li>
            <li>Review one query and its candidate set. Tick checkbox for lookalikes.</li>
            <li>Click <strong>Send Labels &amp; Train</strong> to update feedback/graph/replay and train one cycle.</li>
          </ol>
        </div>

        <h2 className="section-title">SESSION SETTINGS</h2>
        <div className="output-section">
          <div className="settings-grid">
            <label className="setting-item">
              <span>Similarity</span>
              <select value={similarityType} onChange={(e) => setSimilarityType(e.target.value)}>
                <option value="l2">L2</option>
                <option value="ip">Inner Product</option>
                <option value="cosine">Cosine</option>
              </select>
            </label>
            <label className="setting-item">
              <span>Top-K</span>
              <input type="number" value={topK} min={1} max={200} onChange={(e) => setTopK(e.target.value)} />
            </label>
            <label className="setting-item">
              <span>PER alpha</span>
              <input type="number" value={perAlpha} step="0.05" min={0} max={1} onChange={(e) => setPerAlpha(e.target.value)} />
            </label>
            <label className="setting-item">
              <span>PER beta</span>
              <input type="number" value={perBeta} step="0.05" min={0} max={1} onChange={(e) => setPerBeta(e.target.value)} />
            </label>
            <label className="setting-item">
              <span>Batch size</span>
              <input type="number" value={batchSize} min={1} max={512} onChange={(e) => setBatchSize(e.target.value)} />
            </label>
            <label className="setting-item">
              <span>Train steps</span>
              <input type="number" value={trainSteps} min={1} max={20} onChange={(e) => setTrainSteps(e.target.value)} />
            </label>
            <label className="setting-item">
              <span>Min buffer</span>
              <input type="number" value={minBufferSize} min={1} max={5000} onChange={(e) => setMinBufferSize(e.target.value)} />
            </label>
            <label className="setting-item">
              <span>Recent ratio</span>
              <input type="number" value={recentRatio} step="0.05" min={0} max={1} onChange={(e) => setRecentRatio(e.target.value)} />
            </label>
          </div>
          <div className="actions">
            <button
              type="button"
              className="scan-btn scan-btn--active"
              onClick={initSession}
              disabled={loadingInit || loadingSubmit}
            >
              {loadingInit ? 'Initializing…' : 'Initialize Session'}
            </button>
          </div>
        </div>

        {error && <div className="error">{error}</div>}

        <h2 className="section-title">LABELING BATCH</h2>
        <section className="output-section">
          {!batch ? (
            <div className="output-empty">Initialize a session to start labeling.</div>
          ) : (
            <div className="find-result">
              <p className="find-meta">
                Query: {batch.query?.file_name} | Candidates: {batch.candidates?.length ?? 0} | Buffer: {batch.session?.buffer_size ?? 0}
              </p>

              <div className="query-panel">
                <div className="query-title">Current Query</div>
                <div className="query-card">
                  {batch.query?.image_url ? (
                    <img src={toAbsoluteImageUrl(batch.query.image_url)} alt="query" className="query-image" />
                  ) : (
                    <div className="query-image query-image--missing">No image</div>
                  )}
                  <div className="query-meta">
                    <div className="query-name">{batch.query?.file_name}</div>
                    <div className="query-sub">vector_id: {batch.query?.vector_id}</div>
                  </div>
                </div>
              </div>

              <div className="candidate-grid">
                {(batch.candidates ?? []).map((c) => (
                  <div className="candidate-card" key={c.vector_id}>
                    {c.image_url ? (
                      <img src={toAbsoluteImageUrl(c.image_url)} alt={c.file_name} className="candidate-image" />
                    ) : (
                      <div className="candidate-image candidate-image--missing">No image</div>
                    )}
                    <div className="candidate-meta">
                      <div className="candidate-name" title={c.file_name}>{c.file_name}</div>
                      <div className="candidate-sub">
                        id: {c.vector_id} | ret: {c.retrieval_score} | model: {c.model_score}
                      </div>
                      <label className="candidate-label">
                        <input
                          type="checkbox"
                          checked={Boolean(labels[c.vector_id])}
                          onChange={(e) => {
                            setLabels((prev) => ({ ...prev, [c.vector_id]: e.target.checked }))
                          }}
                        />
                        <span>Lookalike</span>
                      </label>
                    </div>
                  </div>
                ))}
              </div>

              <div className="actions actions-row">
                <button
                  type="button"
                  className="scan-btn"
                  onClick={nextBatch}
                  disabled={loadingInit || loadingSubmit}
                >
                  {loadingInit ? 'Loading…' : 'Skip / Next Batch'}
                </button>
                <button
                  type="button"
                  className="scan-btn scan-btn--active"
                  onClick={submitAndTrain}
                  disabled={loadingInit || loadingSubmit}
                >
                  {loadingSubmit ? 'Submitting…' : `Send Labels & Train (${labeledCount} marked)`}
                </button>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
