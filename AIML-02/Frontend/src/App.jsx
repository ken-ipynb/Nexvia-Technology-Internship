
import { useState, useCallback, useEffect } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [activeTab, setActiveTab] = useState("dashboard");

  const [ctFile, setCtFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const [xrayFile, setXrayFile] = useState(null);
  const [xrayPrediction, setXrayPrediction] = useState(null);
  const [xrayLoading, setXrayLoading] = useState(false);
  const [xrayPreviewUrl, setXrayPreviewUrl] = useState("");

  const [error, setError] = useState("");

  const [studyId, setStudyId] = useState("");
  const [reviewer, setReviewer] = useState("");
  const [comment, setComment] = useState("");
  const [reviewLoading, setReviewLoading] = useState(false);
  const [reviewMessage, setReviewMessage] = useState("");

  const [auditLogs, setAuditLogs] = useState([]);
  const [auditLoading, setAuditLoading] = useState(false);
  const [auditError, setAuditError] = useState("");

  useEffect(() => {
    if (!xrayFile) {
      setXrayPreviewUrl("");
      return;
    }

    const imageUrl = URL.createObjectURL(xrayFile);
    setXrayPreviewUrl(imageUrl);

    return () => URL.revokeObjectURL(imageUrl);
  }, [xrayFile]);

  const fetchAuditLogs = useCallback(async () => {
    setAuditLoading(true);
    setAuditError("");

    try {
      const response = await fetch(`${API_URL}/audit-logs`);

      if (!response.ok) {
        throw new Error("Failed to fetch audit logs.");
      }

      const data = await response.json();
      setAuditLogs(data.audit_logs || []);
    } catch (err) {
      setAuditError(err.message || "Unable to load audit history.");
    } finally {
      setAuditLoading(false);
    }
  }, []);

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];

    setCtFile(file || null);
    setPrediction(null);
    setStudyId("");
    setError("");
    setReviewMessage("");
  };

  const handleXrayFileChange = (event) => {
    const file = event.target.files?.[0];

    setXrayFile(file || null);
    setXrayPrediction(null);
    setStudyId("");
    setError("");
    setReviewMessage("");
  };

  const handleCTPrediction = async () => {
    if (!ctFile) {
      setError("Please select a CT scan file first.");
      return;
    }

    setLoading(true);
    setError("");
    setPrediction(null);
    setXrayPrediction(null);
    setStudyId("");
    setReviewMessage("");

    try {
      const formData = new FormData();
      formData.append("file", ctFile);

      const response = await fetch(`${API_URL}/predict/ct`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "CT prediction failed.");
      }

      const positiveVoxels = Number(
        response.headers.get("X-Positive-Voxels") || 0
      );

      const totalVoxels = Number(
        response.headers.get("X-Total-Voxels") || 0
      );

      const segmentationPercentage = Number(
        response.headers.get("X-Segmentation-Percentage") || 0
      );

      const generatedStudyId = `CT-${Date.now()}`;
      setStudyId(generatedStudyId);

      setPrediction({
        positive_voxels: positiveVoxels,
        total_voxels: totalVoxels,
        segmentation_percentage: segmentationPercentage,
      });

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);

      const link = document.createElement("a");
      link.href = url;
      link.download = "lung_segmentation.nii.gz";

      document.body.appendChild(link);
      link.click();
      link.remove();

      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const handleXrayPrediction = async () => {
    if (!xrayFile) {
      setError("Please select an X-ray image first.");
      return;
    }

    setXrayLoading(true);
    setError("");
    setXrayPrediction(null);
    setPrediction(null);
    setStudyId("");
    setReviewMessage("");

    try {
      const formData = new FormData();
      formData.append("file", xrayFile);

      const response = await fetch(`${API_URL}/predict/xray`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "X-ray prediction failed.");
      }

      const data = await response.json();

      setXrayPrediction(data);
      setStudyId(`XR-${Date.now()}`);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setXrayLoading(false);
    }
  };

  const handleReview = async (decision) => {
    if (!studyId) {
      setReviewMessage("Run a prediction before submitting a review.");
      return;
    }

    if (!reviewer.trim()) {
      setReviewMessage("Please enter the reviewer name.");
      return;
    }

    setReviewLoading(true);
    setReviewMessage("");

    try {
      const response = await fetch(`${API_URL}/review`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          study_id: studyId,
          decision,
          reviewer: reviewer.trim(),
          comment: comment.trim(),
        }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(data.detail || "Review submission failed.");
      }

      setReviewMessage(`Review submitted successfully: ${decision}.`);
      setComment("");

      await fetchAuditLogs();
    } catch (err) {
      setReviewMessage(err.message || "Unable to submit review.");
    } finally {
      setReviewLoading(false);
    }
  };

  const renderGradCAM = () => {
    const gradcam = xrayPrediction?.gradcam;

    if (!gradcam) {
      return (
        <p className="error-message">
          Grad-CAM data is not available in the backend response.
        </p>
      );
    }

    return (
      <div className="gradcam-section">
        <h3>Grad-CAM Visualization</h3>

        <p>
          Target disease:{" "}
          <strong>
            {gradcam.target_disease?.replaceAll("_", " ") || "Not available"}
          </strong>
        </p>

        {typeof gradcam.target_score === "number" && (
          <p>
            Target score:{" "}
            <strong>{(gradcam.target_score * 100).toFixed(2)}%</strong>
          </p>
        )}

        <div className="gradcam-grid">
          <div className="gradcam-card">
            <h4>Original X-ray</h4>

            {xrayPreviewUrl ? (
              <img
                src={xrayPreviewUrl}
                alt="Original uploaded chest X-ray"
                className="gradcam-image"
              />
            ) : (
              <p>Original image preview unavailable.</p>
            )}
          </div>

          <div className="gradcam-card">
            <h4>Grad-CAM Heatmap</h4>

            {gradcam.heatmap_base64 ? (
              <img
                src={`data:image/png;base64,${gradcam.heatmap_base64}`}
                alt="Grad-CAM heatmap"
                className="gradcam-image"
              />
            ) : (
              <p>Heatmap image not returned by the backend.</p>
            )}
          </div>

          <div className="gradcam-card">
            <h4>Grad-CAM Overlay</h4>

            {gradcam.overlay_base64 ? (
              <img
                src={`data:image/png;base64,${gradcam.overlay_base64}`}
                alt="Grad-CAM overlay on chest X-ray"
                className="gradcam-image"
              />
            ) : (
              <p>Overlay image not returned by the backend.</p>
            )}
          </div>
        </div>

        <p className="gradcam-note">
          Grad-CAM highlights image regions that influenced the model's
          selected prediction. It is an experimental visualization, not
          clinical evidence.
        </p>
      </div>
    );
  };

  return (
    <div className="app-container">
      <aside className="sidebar">
        <h2 className="logo">MedAI</h2>

        <nav>
          <button
            className={activeTab === "dashboard" ? "active" : ""}
            onClick={() => setActiveTab("dashboard")}
          >
            Dashboard
          </button>

          <button
            className={activeTab === "imaging" ? "active" : ""}
            onClick={() => setActiveTab("imaging")}
          >
            Medical Imaging
          </button>

          <button
            className={activeTab === "predictions" ? "active" : ""}
            onClick={() => setActiveTab("predictions")}
          >
            AI Predictions
          </button>

          <button
            className={activeTab === "audit" ? "active" : ""}
            onClick={() => {
              setActiveTab("audit");
              fetchAuditLogs();
            }}
          >
            Audit History
          </button>
        </nav>
      </aside>

      <main className="main-content">
        <header className="top-header">
          <h1>AI-Assisted Medical Imaging</h1>
          <p>Medical Imaging and Diagnosis System</p>
        </header>

        {activeTab === "dashboard" && (
          <section className="content-section">
            <h2>Dashboard</h2>
            <p>Welcome to the AI-assisted medical imaging system.</p>

            <div className="dashboard-cards">
              <div className="dashboard-card">
                <h3>Medical Imaging</h3>
                <p>Upload and process CT scans or X-ray images.</p>
                <button onClick={() => setActiveTab("imaging")}>
                  Open Imaging
                </button>
              </div>

              <div className="dashboard-card">
                <h3>AI Predictions</h3>
                <p>View AI prediction and segmentation results.</p>
                <button onClick={() => setActiveTab("predictions")}>
                  View Predictions
                </button>
              </div>

              <div className="dashboard-card">
                <h3>Audit History</h3>
                <p>View approval and rejection records.</p>
                <button
                  onClick={() => {
                    setActiveTab("audit");
                    fetchAuditLogs();
                  }}
                >
                  View Audit Logs
                </button>
              </div>
            </div>
          </section>
        )}

        {activeTab === "imaging" && (
          <section className="content-section">
            <h2>Medical Imaging</h2>
            <p>Upload a CT scan or chest X-ray for AI analysis.</p>

            <div className="upload-box">
              <h3>CT Scan Analysis</h3>
              <p>Upload a NIfTI CT scan for 3D lung segmentation.</p>

              <input
                type="file"
                accept=".nii,.nii.gz"
                onChange={handleFileChange}
              />

              {ctFile && <p>Selected file: {ctFile.name}</p>}

              <button
                onClick={handleCTPrediction}
                disabled={loading || xrayLoading || !ctFile}
              >
                {loading ? "Processing CT..." : "Run CT Prediction"}
              </button>
            </div>

            <div className="upload-box">
              <h3>Chest X-ray Analysis</h3>
              <p>Upload a chest X-ray for 14-class disease prediction.</p>

              <input
                type="file"
                accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                onChange={handleXrayFileChange}
              />

              {xrayFile && <p>Selected file: {xrayFile.name}</p>}

              <button
                onClick={handleXrayPrediction}
                disabled={xrayLoading || loading || !xrayFile}
              >
                {xrayLoading ? "Analyzing X-ray..." : "Run X-ray Prediction"}
              </button>
            </div>

            {error && <p className="error-message">{error}</p>}

            {prediction && (
              <div className="result-card">
                <h3>CT Segmentation Completed</h3>

                <p>Positive Voxels: {prediction.positive_voxels}</p>
                <p>Total Voxels: {prediction.total_voxels}</p>
                <p>
                  Segmentation Percentage:{" "}
                  {prediction.segmentation_percentage}%
                </p>
                <p>
                  Study ID: <strong>{studyId}</strong>
                </p>
              </div>
            )}

            {xrayPrediction && (
              <div className="result-card">
                <h3>X-ray Analysis Completed</h3>
                <p>File: {xrayPrediction.filename}</p>
                <p>Model: {xrayPrediction.model}</p>
                <p>Image Size: {xrayPrediction.image_size}</p>
                <p>
                  Study ID: <strong>{studyId}</strong>
                </p>
                <button onClick={() => setActiveTab("predictions")}>
                  View Detailed Predictions
                </button>
              </div>
            )}
          </section>
        )}

        {activeTab === "predictions" && (
          <section className="content-section">
            <h2>AI Predictions</h2>

            {!prediction && !xrayPrediction ? (
              <p>
                No prediction available. Upload a CT scan or X-ray and run
                a prediction first.
              </p>
            ) : (
              <>
                {prediction && (
                  <div className="result-card">
                    <h3>CT Segmentation Result</h3>

                    <p>
                      Study ID: <strong>{studyId}</strong>
                    </p>
                    <p>Positive Voxels: {prediction.positive_voxels}</p>
                    <p>Total Voxels: {prediction.total_voxels}</p>
                    <p>
                      Segmentation Percentage:{" "}
                      {prediction.segmentation_percentage}%
                    </p>
                    <p>The segmentation mask was downloaded as a NIfTI file.</p>
                  </div>
                )}

                {xrayPrediction && (
                  <div className="result-card">
                    <h3>Chest X-ray Prediction Results</h3>

                    <p>File: {xrayPrediction.filename}</p>
                    <p>Model: {xrayPrediction.model}</p>
                    <p>Image Size: {xrayPrediction.image_size}</p>
                    <p>
                      Study ID: <strong>{studyId}</strong>
                    </p>

                    <h4>Disease Prediction Scores</h4>

                    {Object.entries(xrayPrediction.predictions || {}).map(
                      ([disease, score]) => (
                        <div key={disease} className="prediction-item">
                          <div
                            style={{
                              display: "flex",
                              justifyContent: "space-between",
                              marginBottom: "5px",
                            }}
                          >
                            <span>{disease.replaceAll("_", " ")}</span>
                            <strong>{(score * 100).toFixed(2)}%</strong>
                          </div>

                          <div
                            style={{
                              width: "100%",
                              height: "9px",
                              background: "#e5e7eb",
                              borderRadius: "5px",
                              overflow: "hidden",
                              marginBottom: "14px",
                            }}
                          >
                            <div
                              style={{
                                width: `${Math.min(
                                  Math.max(score * 100, 0),
                                  100
                                )}%`,
                                height: "100%",
                                background: "#2563eb",
                                borderRadius: "5px",
                              }}
                            />
                          </div>
                        </div>
                      )
                    )}

                    {renderGradCAM()}

                    <p className="error-message">
                      {xrayPrediction.message}
                    </p>
                  </div>
                )}

                <div className="review-section">
                  <h3>Clinical Review</h3>

                  <p>
                    Enter reviewer details and submit an approval or
                    rejection decision.
                  </p>

                  <div className="review-details">
                    <label htmlFor="reviewer">Reviewer Name</label>

                    <input
                      id="reviewer"
                      type="text"
                      placeholder="Enter reviewer name"
                      value={reviewer}
                      onChange={(e) => setReviewer(e.target.value)}
                    />

                    <label htmlFor="comment">Review Comment</label>

                    <textarea
                      id="comment"
                      placeholder="Enter optional review comments"
                      rows="3"
                      value={comment}
                      onChange={(e) => setComment(e.target.value)}
                    />
                  </div>

                  <div className="review-actions">
                    <button
                      onClick={() => handleReview("Approve")}
                      disabled={reviewLoading}
                    >
                      {reviewLoading ? "Submitting..." : "Approve"}
                    </button>

                    <button
                      onClick={() => handleReview("Reject")}
                      disabled={reviewLoading}
                    >
                      {reviewLoading ? "Submitting..." : "Reject"}
                    </button>
                  </div>

                  {reviewMessage && (
                    <p className="review-message">{reviewMessage}</p>
                  )}
                </div>
              </>
            )}
          </section>
        )}

        {activeTab === "audit" && (
          <section className="content-section">
            <div className="audit-heading">
              <div>
                <h2>Audit History</h2>
                <p>Recorded approval and rejection decisions.</p>
              </div>

              <button
                onClick={fetchAuditLogs}
                disabled={auditLoading}
              >
                {auditLoading ? "Refreshing..." : "Refresh"}
              </button>
            </div>

            {auditError && <p className="error-message">{auditError}</p>}

            {auditLoading ? (
              <p>Loading audit records...</p>
            ) : auditLogs.length === 0 ? (
              <p>No audit records found.</p>
            ) : (
              <div className="audit-table-wrapper">
                <table className="audit-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Study ID</th>
                      <th>Decision</th>
                      <th>Reviewer</th>
                      <th>Comment</th>
                      <th>Timestamp</th>
                    </tr>
                  </thead>

                  <tbody>
                    {auditLogs.map((log) => (
                      <tr key={log.id}>
                        <td>{log.id}</td>
                        <td>{log.study_id}</td>
                        <td
                          className={
                            log.decision === "Approve"
                              ? "decision-approved"
                              : "decision-rejected"
                          }
                        >
                          {log.decision}
                        </td>
                        <td>{log.reviewer}</td>
                        <td>{log.comment || "—"}</td>
                        <td>
                          {log.timestamp
                            ? new Date(log.timestamp).toLocaleString()
                            : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}

        <footer className="app-footer">
          <p>
            Educational prototype only. Not intended for clinical
            diagnosis or treatment decisions.
          </p>
        </footer>
      </main>
    </div>
  );
}

export default App;
