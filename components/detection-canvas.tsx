"use client"

import type React from "react"

import { useEffect, useRef } from "react"

interface DetectionCanvasProps {
  videoRef: React.RefObject<HTMLVideoElement>
  detections: string[]
}

export function DetectionCanvas({ videoRef, detections }: DetectionCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const video = videoRef.current

    if (!canvas || !video) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas size to match video
    canvas.width = video.videoWidth || video.clientWidth
    canvas.height = video.videoHeight || video.clientHeight

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw detection overlay
    if (detections.length > 0) {
      ctx.fillStyle = "rgba(34, 197, 94, 0.2)"
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Draw border
      ctx.strokeStyle = "#22c55e"
      ctx.lineWidth = 4
      ctx.strokeRect(0, 0, canvas.width, canvas.height)
    }
  }, [detections, videoRef])

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
}
