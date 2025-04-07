import { useRef, useState, useEffect, useLayoutEffect } from "react";
import CanvasDraw from "react-canvas-draw";

function App() {
  const [screenDimensions, setScreenDimensions] = useState({
    width: window.innerWidth,
    height: window.innerHeight
  });

  const canvasRef = useRef(null);
  const [brushColor, setBrushColor] = useState("#000000");
  const [brushRadius, setBrushRadius] = useState(4);
  const [prediction, setPrediction] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleClear = () => {
    canvasRef.current.clear();
  };

  const handleUndo = () => {
    canvasRef.current.undo();
  };

  const handleExport = () => {
    const canvas = canvasRef.current.canvas.drawing;
    const ctx = canvas.getContext('2d');

    // Add white background behind existing content
    ctx.globalCompositeOperation = 'destination-over';
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.globalCompositeOperation = 'source-over';

    const dataUrl = canvas.toDataURL("image/png");

    // Convert the data URL to a Blob for uploading
    const imageBlob = dataURLtoBlob(dataUrl);

    // Send the image to the server
    uploadImage(imageBlob);
  };

  const dataURLtoBlob = (dataUrl) => {
    const arr = dataUrl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);

    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }

    return new Blob([u8arr], { type: mime });
  };

  const uploadImage = async (imageBlob) => {
    const formData = new FormData();
    formData.append("image", imageBlob, "drawing.png");

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.prediction) {
        setPrediction(data.prediction);
        setIsModalOpen(true);
      } else {
        console.error("Server error:", data.error);
      }
    } catch (error) {
      console.error("Error sending image:", error);
    }
  };

  // Update screen dimensions on window resize
  useEffect(() => {
    const handleResize = () => {
      setScreenDimensions({
        width: window.innerWidth,
        height: window.innerHeight
      });
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  return (
      <div
          style={{
            fontFamily: "Segoe UI, Roboto, sans-serif",
          }}
      >
        <h2 style={{ textAlign: "center", fontSize: "2rem", marginBottom: "1vw" }}>Text Recognition</h2>

        <CanvasDraw
            ref={canvasRef}
            brushColor={brushColor}
            brushRadius={brushRadius}
            canvasWidth={screenDimensions.width * 0.8}
            canvasHeight={screenDimensions.height * 0.45}
            hideGrid
            style={{
              border: "1px solid #ccc",
              borderRadius: "8px",
              margin: "0 auto",
              display: "block",
              backgroundColor: "#fff",
            }}
        />

        <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              justifyContent: "center",
              padding: "1vw",
              gap: "1vw",
              marginTop: "1.5vw",
            }}
        >
          <Button onClick={() => { setBrushColor("#000000"); setBrushRadius(4); }}>Draw</Button>
          <Button onClick={() => { setBrushColor("#ffffff"); setBrushRadius(32); }}>Erase</Button>
          <Button onClick={handleUndo}>Undo</Button>
          <Button onClick={handleClear}>Clear</Button>
          <Button onClick={handleExport} variant="primary">Send Image</Button>
        </div>

        {isModalOpen && (
            <div
                style={{
                  position: "fixed",
                  top: "50%",
                  left: "50%",
                  transform: "translate(-50%, -50%)",
                  backgroundColor: "rgba(0, 0, 0, 0.9)",
                  color: "white",
                  padding: "20px",
                  borderRadius: "8px",
                  zIndex: 1000,
                  width: "300px",
                  textAlign: "center",
                }}
            >
              <h3>Prediction</h3>
              <p>{prediction}</p>
              <Button onClick={() => setIsModalOpen(false)}>Close</Button>
            </div>
        )}
      </div>
  );
}

function Button({ children, onClick, variant = "default" }) {
  const baseStyle = {
    padding: "10px 16px",
    fontSize: "14px",
    borderRadius: "6px",
    border: "1px solid #ccc",
    backgroundColor: "#f0f0f0",
    color: "#333",
    cursor: "pointer",
    transition: "all 0.2s ease",
  };

  const primaryStyle = {
    backgroundColor: "#0057D9",
    color: "#fff",
    border: "1px solid #0057D9",
  };

  return (
      <button
          onClick={onClick}
          style={{
            ...baseStyle,
            ...(variant === "primary" ? primaryStyle : {}),
          }}
          onMouseEnter={(e) =>
              (e.target.style.backgroundColor = variant === "primary" ? "#004bbf" : "#e2e2e2")
          }
          onMouseLeave={(e) =>
              (e.target.style.backgroundColor = variant === "primary" ? "#0057D9" : "#f0f0f0")
          }
      >
        {children}
      </button>
  );
}

export default App;
