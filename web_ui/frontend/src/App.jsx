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

function Shell({ title, subtitle, backendActive, children }) {
  return (
    <div className="app">
      <div className="content">
        <header className="header">
          <div className="brand">
            <LogoIcon />
            <div>
              <h1 className="brand-title">VeriMed</h1>
              <p className="brand-subtitle">{subtitle}</p>
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

        <nav className="page-nav">
          <a href="/inference" className="page-link">Inference</a>
          <a href="/" className="page-link">Training</a>
          <a href="/replay_buffer" className="page-link">Replay Buffer</a>
          <a href="/ai_feedback_retriever" className="page-link">AI Feedback</a>
        </nav>

        <div className="instructions-panel">
          <h3 className="instructions-title">{title}</h3>
          {children}
        </div>
      </div>
    </div>
  )
}

function ReplayBufferPage({ backendActive }) {
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
    <Shell title="Replay Buffer Monitor" subtitle="Replay Buffer Monitor" backendActive={backendActive}>
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
          </>
        )}
      </div>
    </Shell>
  )
}

function InferencePage({ backendActive }) {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [similarityType, setSimilarityType] = useState('l2')
  const [topK, setTopK] = useState(20)
  const [threshold, setThreshold] = useState(0.5)
  const [result, setResult] = useState(null)

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return undefined
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const runInference = useCallback(async () => {
    if (!file) {
      setError('Upload a query image first.')
      return
    }
    setError(null)
    setLoading(true)
    try {
      const form = new FormData()
      form.append('image', file)
      const params = new URLSearchParams({
        top_k: String(Number(topK)),
        similarity_type: similarityType,
        threshold: String(Number(threshold)),
      })
      const res = await fetch(`${API_BASE}/inference/predict?${params.toString()}`, {
        method: 'POST',
        body: form,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText || 'Inference failed')
      }
      setResult(await res.json())
    } catch (e) {
      setError(e.message || 'Inference failed')
    } finally {
      setLoading(false)
    }
  }, [file, topK, similarityType, threshold])

  return (
    <Shell title="Inference" subtitle="Medication Similarity Inference" backendActive={backendActive}>
      <ol className="instructions-list">
        <li>Upload a query image.</li>
        <li>Choose retrieval mode, top-k, and threshold.</li>
        <li>Click <strong>Run Inference</strong> to retrieve candidates and classify by assessor score.</li>
      </ol>

      <h2 className="section-title">INFERENCE SETTINGS</h2>
      <div className="output-section">
        <div className={`upload-zone ${previewUrl ? 'has-image' : ''}`}>
          <label className="upload-label">
            <input
              className="upload-input"
              type="file"
              accept="image/*"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            {previewUrl ? (
              <img src={previewUrl} alt="query preview" className="upload-preview" />
            ) : (
              <>
                <div className="upload-text">Click to upload a query image</div>
                <div className="upload-hint">JPG, PNG, WEBP, BMP, TIFF</div>
              </>
            )}
          </label>
        </div>
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
            <span>Threshold</span>
            <input
              type="number"
              value={threshold}
              min={0}
              max={1}
              step={0.01}
              onChange={(e) => setThreshold(e.target.value)}
            />
          </label>
        </div>
        <div className="actions">
          <button type="button" className="scan-btn scan-btn--active" onClick={runInference} disabled={loading}>
            {loading ? 'Running…' : 'Run Inference'}
          </button>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      <h2 className="section-title">RESULTS</h2>
      <section className="output-section">
        {!result ? (
          <div className="output-empty">Upload a query image and run inference.</div>
        ) : (
          <div className="find-result">
            <p className="find-meta">
              Similarity: {result.meta?.similarity_type} | Top-K: {result.meta?.top_k} | Threshold: {result.meta?.threshold}
            </p>
            <div className="query-panel">
              <div className="query-title">Uploaded Query</div>
              <div className="query-card">
                {previewUrl ? (
                  <img src={previewUrl} alt="uploaded query" className="query-image" />
                ) : (
                  <div className="query-image query-image--missing">No image</div>
                )}
                <div className="query-meta">
                  <div className="query-name">{file?.name ?? 'Uploaded query'}</div>
                  <div className="query-sub">Top-k candidates were scored by the assessor.</div>
                </div>
              </div>
            </div>
            <div className="result-columns">
              <ResultList title="Lookalikes" items={result.lookalikes ?? []} />
              <ResultList title="Non-lookalikes" items={result.non_lookalikes ?? []} />
            </div>
          </div>
        )}
      </section>
    </Shell>
  )
}

function ResultList({ title, items }) {
  return (
    <div className={`result-list ${title === 'Lookalikes' ? 'lookalikes' : 'not-lookalikes'}`}>
      <h3>{title}</h3>
      <div className="result-list-body">
        {items.length === 0 ? (
          <div className="output-empty output-empty--compact">No items.</div>
        ) : (
          <div className="candidate-grid candidate-grid--scroll">
            {items.map((c) => (
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
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function AIFeedbackPage({ backendActive }) {
  const [loadingQueries, setLoadingQueries] = useState(false)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [error, setError] = useState(null)
  const [queries, setQueries] = useState([])
  const [selectedQueryId, setSelectedQueryId] = useState('')
  const [detail, setDetail] = useState(null)

  const loadQueries = useCallback(async () => {
    setError(null)
    setLoadingQueries(true)
    try {
      const res = await fetch(`${API_BASE}/ai_feedback/queries?limit=100`)
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText || 'Failed to load AI feedback queries')
      }
      const data = await res.json()
      const nextQueries = data.queries ?? []
      setQueries(nextQueries)
      setSelectedQueryId((prev) => {
        if (prev && nextQueries.some((q) => String(q.query_vector_id) === String(prev))) return prev
        return nextQueries[0] ? String(nextQueries[0].query_vector_id) : ''
      })
    } catch (e) {
      setError(e.message || 'Failed to load AI feedback queries')
    } finally {
      setLoadingQueries(false)
    }
  }, [])

  const loadDetail = useCallback(async (queryId) => {
    if (!queryId) {
      setDetail(null)
      return
    }
    setError(null)
    setLoadingDetail(true)
    try {
      const res = await fetch(`${API_BASE}/ai_feedback/by_query?query_vector_id=${Number(queryId)}`)
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText || 'Failed to load AI feedback detail')
      }
      setDetail(await res.json())
    } catch (e) {
      setError(e.message || 'Failed to load AI feedback detail')
    } finally {
      setLoadingDetail(false)
    }
  }, [])

  useEffect(() => {
    loadQueries()
  }, [loadQueries])

  useEffect(() => {
    loadDetail(selectedQueryId)
  }, [selectedQueryId, loadDetail])

  return (
    <Shell title="AI Feedback Retriever" subtitle="Batch AI Feedback Viewer" backendActive={backendActive}>
      <ol className="instructions-list">
        <li>Choose a query image imported from AI batch results.</li>
        <li>Review which candidates the AI marked as lookalike or non-lookalike.</li>
        <li>Check the stored reasoning for each candidate pair.</li>
      </ol>

      <h2 className="section-title">QUERY SELECTOR</h2>
      <div className="output-section">
        <div className="actions actions-row actions-row--query">
          <label className="setting-item">
            <span>Imported query</span>
            <select value={selectedQueryId} onChange={(e) => setSelectedQueryId(e.target.value)} disabled={loadingQueries || queries.length === 0}>
              {queries.length === 0 ? (
                <option value="">No imported AI feedback yet</option>
              ) : (
                queries.map((q) => (
                  <option key={q.query_vector_id} value={q.query_vector_id}>
                    {q.query_image_name}
                  </option>
                ))
              )}
            </select>
          </label>
          <button type="button" className="scan-btn scan-btn--active" onClick={loadQueries} disabled={loadingQueries || loadingDetail}>
            {loadingQueries ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
        {queries.length > 0 && selectedQueryId && (
          <p className="find-meta">
            {(() => {
              const selected = queries.find((q) => String(q.query_vector_id) === String(selectedQueryId))
              if (!selected) return null
              return `Lookalikes: ${selected.lookalike_count} | Non-lookalikes: ${selected.non_lookalike_count} | Total: ${selected.total_count}`
            })()}
          </p>
        )}
      </div>

      {error && <div className="error">{error}</div>}

      <h2 className="section-title">AI FEEDBACK RESULTS</h2>
      <section className="output-section">
        {!selectedQueryId ? (
          <div className="output-empty">Import AI feedback first, then choose a query.</div>
        ) : loadingDetail ? (
          <div className="output-empty">Loading AI feedback…</div>
        ) : !detail ? (
          <div className="output-empty">No AI feedback detail loaded.</div>
        ) : (
          <div className="find-result">
            <p className="find-meta">
              Query: {detail.query?.file_name} | Lookalikes: {detail.meta?.lookalike_count} | Non-lookalikes: {detail.meta?.non_lookalike_count}
            </p>
            <div className="query-panel">
              <div className="query-title">Imported Query</div>
              <div className="query-card">
                {detail.query?.image_url ? (
                  <img src={toAbsoluteImageUrl(detail.query.image_url)} alt="ai feedback query" className="query-image" />
                ) : (
                  <div className="query-image query-image--missing">No image</div>
                )}
                <div className="query-meta">
                  <div className="query-name">{detail.query?.file_name}</div>
                  <div className="query-sub">vector_id: {detail.query?.vector_id}</div>
                  <div className="query-sub">batch count: {detail.meta?.batch_ids?.length ?? 0}</div>
                </div>
              </div>
            </div>
            <div className="result-columns">
              <AIFeedbackResultList title="Lookalikes" items={detail.lookalikes ?? []} />
              <AIFeedbackResultList title="Non-lookalikes" items={detail.non_lookalikes ?? []} />
            </div>
          </div>
        )}
      </section>
    </Shell>
  )
}

function AIFeedbackResultList({ title, items }) {
  return (
    <div className={`result-list ${title === 'Lookalikes' ? 'lookalikes' : 'not-lookalikes'}`}>
      <h3>{title}</h3>
      <div className="result-list-body">
        {items.length === 0 ? (
          <div className="output-empty output-empty--compact">No items.</div>
        ) : (
          <div className="candidate-grid candidate-grid--scroll">
            {items.map((item, idx) => (
              <div className="candidate-card candidate-card--reasoning" key={`${item.vector_id}-${idx}`}>
                {item.image_url ? (
                  <img src={toAbsoluteImageUrl(item.image_url)} alt={item.file_name} className="candidate-image" />
                ) : (
                  <div className="candidate-image candidate-image--missing">No image</div>
                )}
                <div className="candidate-meta">
                  <div className="candidate-name" title={item.file_name}>{item.file_name}</div>
                  <div className="candidate-sub">
                    id: {item.vector_id} | batch: {item.batch_id}
                  </div>
                  {item.identical && (
                    <div className="candidate-flag">Identical</div>
                  )}
                  <div className="candidate-reasoning">{item.reasoning}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function OnlineTrainingPage({ backendActive }) {
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
  const [loadReplayFromDb, setLoadReplayFromDb] = useState(false)
  const [hydratedCount, setHydratedCount] = useState(0)

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
          hydrate_from_feedback: loadReplayFromDb,
        }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText || 'Session init failed')
      }
      const data = await res.json()
      setHydratedCount(Number(data.hydrated ?? 0))
      setBatch(data.batch)
      setLabels(buildDefaultLabels(data.batch?.candidates))
    } catch (e) {
      setError(e.message || 'Session init failed')
    } finally {
      setLoadingInit(false)
    }
  }, [similarityType, topK, perAlpha, perBeta, batchSize, trainSteps, minBufferSize, recentRatio, loadReplayFromDb])

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
    <Shell title="Online Labeling + Training Loop" subtitle="Medication Detection Verification System" backendActive={backendActive}>
      <ol className="instructions-list">
        <li>Set retrieval mode and top-k, then click <strong>Initialize Session</strong>.</li>
        <li>Review one query and its candidate set. Tick checkbox for lookalikes.</li>
        <li>Click <strong>Send Labels &amp; Train</strong> to update feedback/graph/replay and train one cycle.</li>
      </ol>

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
        <label className="session-toggle">
          <input
            type="checkbox"
            checked={loadReplayFromDb}
            onChange={(e) => setLoadReplayFromDb(e.target.checked)}
          />
          <span>Load replay buffer from DB before session start</span>
        </label>
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
            {loadReplayFromDb && (
              <p className="find-meta">Hydrated from DB: {hydratedCount}</p>
            )}

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
    </Shell>
  )
}

export default function App() {
  const pathname = typeof window !== 'undefined' ? window.location.pathname : '/'
  const [backendActive, setBackendActive] = useState(null)

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

  if (pathname === '/replay_buffer') {
    return <ReplayBufferPage backendActive={backendActive} />
  }
  if (pathname === '/ai_feedback_retriever') {
    return <AIFeedbackPage backendActive={backendActive} />
  }
  if (pathname === '/inference') {
    return <InferencePage backendActive={backendActive} />
  }
  return <OnlineTrainingPage backendActive={backendActive} />
}
