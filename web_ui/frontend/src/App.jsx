import { useState, useCallback, useRef, useEffect } from 'react'
import { BriefcaseMedical, ImageUp } from 'lucide-react'
import './App.css'

const API_BASE = import.meta.env.DEV ? '/api' : 'http://127.0.0.1:8000'

/** Strip "embeddings / " or "embeddings/" prefix for display. */
function displayName(name) {
  if (!name || typeof name !== 'string') return name ?? ''
  return name.replace(/^embeddings\s*\/\s*/i, '').trim() || name
}

const LogoIcon = () => (
  <BriefcaseMedical size={32} className="logo-icon" aria-hidden />
)

const SearchIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" className="btn-icon">
    <circle cx="11" cy="11" r="7"/>
    <path d="M20 20l-4.5-4.5"/>
  </svg>
)

const ImagePlaceholderIcon = () => (
  <ImageUp size={56} className="upload-icon" aria-hidden />
)

export default function App() {
  const [queryImage, setQueryImage] = useState(null)
  const [findResult, setFindResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [sortBy, setSortBy] = useState('score') // 'score' | 'alphabetical'
  const [backendActive, setBackendActive] = useState(null) // null = checking, true/false = known
  const progressIntervalRef = useRef(null)

  useEffect(() => {
    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current)
      }
    }
  }, [])

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

  const setFile = useCallback((file) => {
    if (file && file.type.startsWith('image/')) {
      setQueryImage(file)
      setFindResult(null)
      setError(null)
    }
  }, [])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer?.files?.[0]
    setFile(file || null)
  }, [setFile])

  const onDragOver = useCallback((e) => {
    e.preventDefault()
    setDragOver(true)
  }, [])

  const onDragLeave = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
  }, [])

  const handleAnalyze = async () => {
    if (!queryImage) {
      setError('Please select an image.')
      return
    }
    setError(null)
    setFindResult(null)
    setProgress(0)
    setLoading(true)
    // Simulated progress 0 -> 90% while request is in flight
    let p = 0
    progressIntervalRef.current = setInterval(() => {
      p = Math.min(p + 4, 90)
      setProgress(p)
      if (p >= 90 && progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current)
        progressIntervalRef.current = null
      }
    }, 120)
    try {
      const form = new FormData()
      form.append('query_image', queryImage)
      const res = await fetch(`${API_BASE}/find_lookalikes`, {
        method: 'POST',
        body: form,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText || 'Scan failed')
      }
      const data = await res.json()
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current)
        progressIntervalRef.current = null
      }
      setProgress(100)
      setFindResult(data)
    } catch (e) {
      setError(e.message || 'Request failed')
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current)
        progressIntervalRef.current = null
      }
      setProgress(0)
    } finally {
      setLoading(false)
      setTimeout(() => setProgress(0), 400)
    }
  }

  const urlQuery = queryImage ? URL.createObjectURL(queryImage) : null

  const sortItems = (items, byScoreDesc = true) => {
    if (!items?.length) return []
    const list = [...items]
    if (sortBy === 'alphabetical') {
      list.sort((a, b) => (a.name ?? '').localeCompare(b.name ?? '', undefined, { sensitivity: 'base' }))
      return list
    }
    if (!byScoreDesc) return list
    list.sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
    return list
  }

  const lookalikeItems = sortItems(findResult?.lookalikes ?? [], true)
  const notLookalikeItems = sortItems(findResult?.not_lookalikes ?? [], true)

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
          <h3 className="instructions-title">Instructions</h3>
          <ol className="instructions-list">
            <li>Upload a photo of the medication you want to verify (PNG or JPG, up to 10MB).</li>
            <li>Click <strong>Scan Database</strong> to search for lookalikes in the catalog.</li>
            <li>Review the results in the Output section below.</li>
          </ol>
        </div>

        <h2 className="section-title">UPLOAD QUERY MEDICATION</h2>

        <div
          className={`upload-zone ${dragOver ? 'drag-over' : ''} ${urlQuery ? 'has-image' : ''}`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
        >
          <label className="upload-label">
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp,image/gif"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="upload-input"
            />
            {urlQuery ? (
              <img src={urlQuery} alt="Upload preview" className="upload-preview" />
            ) : (
              <>
                <ImagePlaceholderIcon />
                <span className="upload-text">Click or drag image here</span>
                <span className="upload-hint">PNG, JPG up to 10MB</span>
              </>
            )}
          </label>
        </div>

        <div className="actions">
          <button
            type="button"
            className={`scan-btn ${queryImage ? 'scan-btn--active' : ''}`}
            onClick={handleAnalyze}
            disabled={loading || !queryImage}
          >
            <SearchIcon />
            <span>{loading ? 'Scanning…' : 'Scan Database'}</span>
          </button>
        </div>

        {error && <div className="error">{error}</div>}

        {loading && (
          <div className="scan-progress">
            <div className="scan-progress-bar">
              <div className="scan-progress-fill" style={{ width: `${progress}%` }} />
            </div>
            <span className="scan-progress-text">{progress}% — Scanning database…</span>
          </div>
        )}

        <section className="output-section">
          <h3 className="output-title">Output</h3>
          {findResult ? (
            <div className="find-result">
              <p className="find-meta">
                Compared to {findResult.total_candidates} candidates (threshold: {findResult.threshold}).
                {findResult.message && ` ${findResult.message}`}
              </p>
              <div className="output-sort">
                <span className="output-sort-label">Sort:</span>
                <label className="output-sort-option">
                  <input
                    type="radio"
                    name="sortBy"
                    checked={sortBy === 'score'}
                    onChange={() => setSortBy('score')}
                  />
                  <span>By score</span>
                </label>
                <label className="output-sort-option">
                  <input
                    type="radio"
                    name="sortBy"
                    checked={sortBy === 'alphabetical'}
                    onChange={() => setSortBy('alphabetical')}
                  />
                  <span>A–Z</span>
                </label>
              </div>
              <div className="result-columns">
                <div className="result-list lookalikes">
                  <h3>Lookalikes (score ≥ 0.5) <span className="result-count">({lookalikeItems.length})</span></h3>
                  <ul>
                    {lookalikeItems.map((item, i) => (
                      <li key={i}>
                        <span className="name" title={item.name}>{displayName(item.name)}</span>
                        <span className="score" title="Assessor score (lookalike probability)">{item.score}</span>
                      </li>
                    ))}
                    {lookalikeItems.length === 0 && (
                      <li className="empty">None</li>
                    )}
                  </ul>
                </div>
                <div className="result-list not-lookalikes">
                  <h3>Non-lookalikes (score &lt; 0.5) <span className="result-count">({notLookalikeItems.length})</span></h3>
                  <ul>
                    {notLookalikeItems.map((item, i) => (
                      <li key={i}>
                        <span className="name" title={item.name}>{displayName(item.name)}</span>
                        <span className="score" title="Assessor score (lookalike probability)">{item.score}</span>
                      </li>
                    ))}
                    {notLookalikeItems.length === 0 && (
                      <li className="empty">None</li>
                    )}
                  </ul>
                </div>
              </div>
            </div>
          ) : loading ? (
            <div className="output-empty">Scanning database… {progress}%</div>
          ) : (
            <div className="output-empty">No results yet. Upload an image and click Scan Database.</div>
          )}
        </section>
      </div>
    </div>
  )
}
