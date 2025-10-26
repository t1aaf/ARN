"use client"

import type React from "react"

import { useState, useEffect, useRef, useCallback } from "react"
import * as cocoSsd from "@tensorflow-models/coco-ssd"
import "@tensorflow/tfjs"

interface Detection {
  class: string
  score: number
  bbox: [number, number, number, number]
}

export function useObjectDetection(videoRef: React.RefObject<HTMLVideoElement>, isSpeaking = false) {
  const [detections, setDetections] = useState<string[]>([])
  const [isModelLoading, setIsModelLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null)
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const lastAnnouncementRef = useRef<string>("")
  const lastAnnouncementTimeRef = useRef<number>(0)
  const lastDetectionRef = useRef<string>("")

  // Load model on mount
  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log("[v0] Loading COCO-SSD model...")
        const model = await cocoSsd.load()
        modelRef.current = model
        setIsModelLoading(false)
        setError(null)
        console.log("[v0] Model loaded successfully")
      } catch (error) {
        console.error("[v0] Error loading model:", error)
        setError("Failed to load detection model")
        setIsModelLoading(false)
      }
    }

    loadModel()

    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current)
      }
    }
  }, [])

  const estimateDistance = (
    bbox: [number, number, number, number],
    videoWidth: number,
    objectClass = "person",
  ): number => {
    try {
      const [, , width, height] = bbox

      // Expected heights of common objects in meters
      const expectedHeights: Record<string, number> = {
        person: 1.7,
        car: 1.5,
        truck: 2.5,
        motorcycle: 1.2,
        bus: 3.0,
        dog: 0.6,
        cat: 0.3,
        bicycle: 1.0,
        chair: 0.8,
        door: 2.0,
        tree: 5.0,
        pole: 0.3,
        traffic_cone: 0.7,
        bench: 0.5,
        wall: 2.0,
      }

      const expectedHeight = expectedHeights[objectClass] || 1.0

      // Assume 60-degree vertical field of view for most cameras
      const verticalFOV = 60 * (Math.PI / 180)
      const focalLength = videoWidth / (2 * Math.tan(verticalFOV / 2))

      // Calculate distance using pinhole camera model
      const pixelHeight = height
      const distance = (expectedHeight * focalLength) / pixelHeight

      // Clamp distance to reasonable range (0.3m to 20m)
      return Math.max(0.3, Math.min(20, Number.parseFloat(distance.toFixed(1))))
    } catch (error) {
      console.error("[v0] Distance estimation error:", error)
      return 1.0
    }
  }

  const getPosition = (bbox: [number, number, number, number], videoWidth: number): string => {
    try {
      const [x, , width] = bbox
      const centerX = x + width / 2
      const relativePosition = centerX / videoWidth

      if (relativePosition < 0.33) return "left"
      if (relativePosition > 0.67) return "right"
      return "center"
    } catch (error) {
      console.error("[v0] Position calculation error:", error)
      return "center"
    }
  }

  const detectWall = useCallback(
    (video: HTMLVideoElement): { hasWall: boolean; distance: number; position: string } | null => {
      try {
        if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
          return null
        }

        const canvas = document.createElement("canvas")
        const ctx = canvas.getContext("2d")
        if (!ctx) return null

        canvas.width = video.videoWidth
        canvas.height = video.videoHeight

        try {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        } catch (drawError) {
          console.warn("[v0] Canvas draw error (likely CORS):", drawError)
          return null
        }

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
        const data = imageData.data

        // Analyze the center region for wall characteristics
        const centerX = canvas.width / 2
        const centerY = canvas.height / 2
        const regionSize = Math.min(canvas.width, canvas.height) / 3

        let edgeCount = 0
        let totalBrightness = 0
        let pixelCount = 0

        // Sobel edge detection in center region
        for (
          let y = Math.max(0, centerY - regionSize / 2);
          y < Math.min(canvas.height, centerY + regionSize / 2);
          y++
        ) {
          for (
            let x = Math.max(0, centerX - regionSize / 2);
            x < Math.min(canvas.width, centerX + regionSize / 2);
            x++
          ) {
            const idx = (y * canvas.width + x) * 4
            const r = data[idx]
            const g = data[idx + 1]
            const b = data[idx + 2]
            const brightness = (r + g + b) / 3

            totalBrightness += brightness
            pixelCount++

            // Check for edges (high contrast)
            if (x > 0 && x < canvas.width - 1 && y > 0 && y < canvas.height - 1) {
              const leftIdx = (y * canvas.width + (x - 1)) * 4
              const rightIdx = (y * canvas.width + (x + 1)) * 4
              const topIdx = ((y - 1) * canvas.width + x) * 4
              const bottomIdx = ((y + 1) * canvas.width + x) * 4

              const leftBrightness = (data[leftIdx] + data[leftIdx + 1] + data[leftIdx + 2]) / 3
              const rightBrightness = (data[rightIdx] + data[rightIdx + 1] + data[rightIdx + 2]) / 3
              const topBrightness = (data[topIdx] + data[topIdx + 1] + data[topIdx + 2]) / 3
              const bottomBrightness = (data[bottomIdx] + data[bottomIdx + 1] + data[bottomIdx + 2]) / 3

              const edgeMagnitude =
                Math.abs(leftBrightness - rightBrightness) + Math.abs(topBrightness - bottomBrightness)
              if (edgeMagnitude > 30) edgeCount++
            }
          }
        }

        if (pixelCount === 0) return null

        const avgBrightness = totalBrightness / pixelCount
        const edgeDensity = edgeCount / pixelCount

        // Wall detection: low edge density (flat surface) and moderate brightness
        if (edgeDensity < 0.1 && avgBrightness > 50) {
          // Closer walls appear brighter and more uniform
          let distance = 10

          if (avgBrightness > 220) distance = 0.5
          else if (avgBrightness > 210) distance = 0.7
          else if (avgBrightness > 200) distance = 1.0
          else if (avgBrightness > 190) distance = 1.3
          else if (avgBrightness > 180) distance = 1.6
          else if (avgBrightness > 170) distance = 2.0
          else if (avgBrightness > 160) distance = 2.5
          else if (avgBrightness > 150) distance = 3.0
          else if (avgBrightness > 140) distance = 3.5
          else if (avgBrightness > 130) distance = 4.0
          else if (avgBrightness > 120) distance = 4.5
          else if (avgBrightness > 110) distance = 5.0
          else if (avgBrightness > 100) distance = 5.5
          else distance = 6.0

          // Determine position based on brightness distribution
          const leftRegion = canvas.width / 3
          const rightRegion = (canvas.width * 2) / 3

          let leftBrightness = 0,
            centerBrightness = 0,
            rightBrightness = 0
          let leftCount = 0,
            centerCount = 0,
            rightCount = 0

          for (
            let y = Math.max(0, centerY - regionSize / 2);
            y < Math.min(canvas.height, centerY + regionSize / 2);
            y++
          ) {
            for (
              let x = Math.max(0, centerX - regionSize / 2);
              x < Math.min(canvas.width, centerX + regionSize / 2);
              x++
            ) {
              const idx = (y * canvas.width + x) * 4
              const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3

              if (x < leftRegion) {
                leftBrightness += brightness
                leftCount++
              } else if (x < rightRegion) {
                centerBrightness += brightness
                centerCount++
              } else {
                rightBrightness += brightness
                rightCount++
              }
            }
          }

          leftBrightness /= leftCount || 1
          centerBrightness /= centerCount || 1
          rightBrightness /= rightCount || 1

          let position = "center"
          if (leftBrightness > centerBrightness && leftBrightness > rightBrightness) position = "left"
          else if (rightBrightness > centerBrightness && rightBrightness > leftBrightness) position = "right"

          return {
            hasWall: true,
            distance: Number.parseFloat(distance.toFixed(1)),
            position,
          }
        }

        return null
      } catch (error) {
        console.error("[v0] Wall detection error:", error)
        return null
      }
    },
    [],
  )

  const detectObjects = useCallback(async () => {
    if (isSpeaking) {
      return
    }

    const video = videoRef.current
    const model = modelRef.current

    if (!video || !model || video.readyState !== 4) return

    try {
      const predictions = await model.detect(video)

      // Priority 1: Detect people (highest priority)
      const people = predictions.filter((p) => p.class === "person")

      if (people.length > 0) {
        const closestPerson = people.reduce((closest, current) => {
          const closestDist = estimateDistance(closest.bbox, video.videoWidth, "person")
          const currentDist = estimateDistance(current.bbox, video.videoWidth, "person")
          return currentDist < closestDist ? current : closest
        })

        const distance = estimateDistance(closestPerson.bbox, video.videoWidth, "person")
        const position = getPosition(closestPerson.bbox, video.videoWidth)

        let announcement = ""
        if (distance < 2) {
          announcement = `Very close person ${position} about ${distance} meters`
        } else {
          announcement = `Person ${position} about ${distance} meters away`
        }

        const now = Date.now()
        if (announcement !== lastDetectionRef.current || now - lastAnnouncementTimeRef.current > 4000) {
          setDetections([announcement])
          lastDetectionRef.current = announcement
          lastAnnouncementTimeRef.current = now
        }
        return
      }

      // Priority 2: Detect vehicles (cars, trucks, motorcycles)
      const vehicles = predictions.filter((p) => ["car", "truck", "motorcycle", "bus"].includes(p.class))

      if (vehicles.length > 0) {
        const closestVehicle = vehicles.reduce((closest, current) => {
          const closestDist = estimateDistance(closest.bbox, video.videoWidth, current.class)
          const currentDist = estimateDistance(current.bbox, video.videoWidth, current.class)
          return currentDist < closestDist ? current : closest
        })

        const distance = estimateDistance(closestVehicle.bbox, video.videoWidth, closestVehicle.class)
        const position = getPosition(closestVehicle.bbox, video.videoWidth)

        const announcement = `${closestVehicle.class} ${position} about ${distance} meters away`

        const now = Date.now()
        if (announcement !== lastDetectionRef.current || now - lastAnnouncementTimeRef.current > 4000) {
          setDetections([announcement])
          lastDetectionRef.current = announcement
          lastAnnouncementTimeRef.current = now
        }
        return
      }

      // Priority 3: Detect other obstacles (furniture, hazards, structural objects, etc.)
      const obstacles = predictions.filter((p) =>
        [
          // Furniture and indoor objects
          "chair",
          "couch",
          "bed",
          "dining table",
          "door",
          "tv",
          "laptop",
          "bottle",
          "cup",
          "backpack",
          "handbag",
          "suitcase",
          // Hazard objects
          "umbrella",
          "sports ball",
          "baseball bat",
          "tennis racket",
          "potted plant",
          "plant",
          "traffic cone",
          "bench",
          "railing",
          "fence",
          "pole",
          "skateboard",
          "bicycle",
          "motorcycle helmet",
          // Animals
          "dog",
          "cat",
          "bird",
          "teddy bear",
          // Outdoor hazards
          "kite",
          "frisbee",
          "stairs",
          "step",
          "ramp",
          "escalator",
          "elevator",
          "pillar",
          "column",
          "barrier",
          "gate",
          "sign",
          "lamp",
          "street light",
          "hydrant",
          "mailbox",
          "trash can",
          "dumpster",
          "wall",
          "building",
          "bridge",
          "tunnel",
          "curb",
          "manhole",
          "grate",
          "bollard",
          "post",
          "tree",
          "bush",
          "rock",
          "log",
          "branch",
          "stick",
        ].includes(p.class),
      )

      if (obstacles.length > 0) {
        const closestObstacle = obstacles.reduce((closest, current) => {
          const closestDist = estimateDistance(closest.bbox, video.videoWidth, current.class)
          const currentDist = estimateDistance(current.bbox, video.videoWidth, current.class)
          return currentDist < closestDist ? current : closest
        })

        const distance = estimateDistance(closestObstacle.bbox, video.videoWidth, closestObstacle.class)
        const position = getPosition(closestObstacle.bbox, video.videoWidth)

        const announcement = `${closestObstacle.class} ${position} about ${distance} meters away`

        const now = Date.now()
        if (announcement !== lastDetectionRef.current || now - lastAnnouncementTimeRef.current > 4000) {
          setDetections([announcement])
          lastDetectionRef.current = announcement
          lastAnnouncementTimeRef.current = now
        }
        return
      }

      // Priority 4: Detect walls
      const wallDetection = detectWall(video)
      if (wallDetection && wallDetection.hasWall) {
        const announcement = `Wall ${wallDetection.position} about ${wallDetection.distance} meters ahead`

        const now = Date.now()
        if (announcement !== lastDetectionRef.current || now - lastAnnouncementTimeRef.current > 4000) {
          setDetections([announcement])
          lastDetectionRef.current = announcement
          lastAnnouncementTimeRef.current = now
        }
        return
      }

      // No objects detected - announce clear path less frequently
      const now = Date.now()
      if (now - lastAnnouncementTimeRef.current > 6000) {
        const announcement = "Clear path ahead"
        if (announcement !== lastDetectionRef.current) {
          setDetections([announcement])
          lastDetectionRef.current = announcement
          lastAnnouncementTimeRef.current = now
        }
      }
    } catch (error) {
      console.error("[v0] Detection error:", error)
      setError("Detection error occurred")
    }
  }, [videoRef, isSpeaking, detectWall])

  const startDetection = useCallback(async () => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current)
    }

    detectionIntervalRef.current = setInterval(() => {
      detectObjects()
    }, 1000)
  }, [detectObjects])

  const stopDetection = useCallback(() => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current)
      detectionIntervalRef.current = null
    }
    setDetections([])
    lastDetectionRef.current = ""
  }, [])

  return {
    detections: Array.isArray(detections) ? detections : [],
    isModelLoading,
    error,
    startDetection,
    stopDetection,
  }
}
