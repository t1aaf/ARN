"use client";

import { useEffect, useRef, useState } from "react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { DetectionCanvas } from "@/components/detection-canvas";
import { useObjectDetection } from "@/hooks/use-object-detection";
import { useSpeech } from "@/hooks/use-speech";
import { Eye, Mic, Camera, Waves } from "lucide-react";

export default function ARGuidePage() {
  const videoRef = useRef<HTMLVideoElement>(
    null
  ) as React.RefObject<HTMLVideoElement>;
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isSpeechEnabled, setIsSpeechEnabled] = useState(true);

  const { speak, isSpeaking, stopSpeaking, testSpeech } = useSpeech();
  const { detections, isModelLoading, startDetection, stopDetection } =
    useObjectDetection(videoRef, isSpeaking);

  useEffect(() => {
    const initialize = async () => {
      try {
        setError(null);

        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1920, max: 1920 },
            height: { ideal: 1080, max: 1080 },
            aspectRatio: { ideal: 16 / 9 },
          },
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          videoRef.current.setAttribute("playsinline", "true");
          videoRef.current.setAttribute("webkit-playsinline", "true");
          await videoRef.current.play();
          setStream(mediaStream);
          setIsActive(true);
          await startDetection();

          testSpeech();
          setIsSpeechEnabled(true);
        }
      } catch (err) {
        console.error("[v0] Initialization error:", err);
        setError(
          "Unable to access camera or speech. Please grant permissions and reload."
        );
      }
    };

    initialize();

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      stopDetection();
      stopSpeaking();
    };
  }, []);

  useEffect(() => {
    if (!isSpeechEnabled || !isActive || detections.length === 0) return;

    const announcement = detections[0];
    if (announcement) {
      speak(announcement);
    }
  }, [detections, isSpeechEnabled, isActive, speak]);

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-background via-background to-secondary overflow-hidden">
      <header className="absolute top-0 left-0 right-0 z-20 p-4 md:p-6 bg-gradient-to-b from-background/80 to-transparent backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 md:w-12 md:h-12 rounded-full bg-primary/20 border border-primary/30">
            <Eye className="w-5 h-5 md:w-6 md:h-6 text-primary" />
          </div>
          <div>
            <h1 className="text-xl md:text-2xl lg:text-3xl font-bold text-foreground tracking-tight">
              Vision Guide
            </h1>
            <p className="text-xs md:text-sm text-muted-foreground">
              AI-powered navigation assistant
            </p>
          </div>
        </div>
      </header>

      <main className="absolute inset-0 flex items-center justify-center p-4 pt-24 pb-32 md:pt-28 md:pb-36">
        {error && (
          <Alert
            variant="destructive"
            className="absolute top-24 left-4 right-4 z-30 md:left-6 md:right-6"
          >
            <AlertDescription className="text-sm md:text-base">
              {error}
            </AlertDescription>
          </Alert>
        )}

        {isModelLoading && (
          <Alert className="absolute top-24 left-4 right-4 z-30 md:left-6 md:right-6 bg-card/90 backdrop-blur-sm border-primary/30">
            <AlertDescription className="text-sm md:text-base flex items-center gap-2">
              <Waves className="w-4 h-4 animate-pulse text-primary" />
              Loading AI detection model...
            </AlertDescription>
          </Alert>
        )}

        <div className="relative w-full h-full max-w-6xl rounded-2xl md:rounded-3xl overflow-hidden shadow-2xl border border-border/50">
          <video
            ref={videoRef}
            className="absolute inset-0 w-full h-full object-cover"
            playsInline
            muted
            autoPlay
          />
          {isActive && (
            <DetectionCanvas videoRef={videoRef} detections={detections} />
          )}

          {!isActive && (
            <div className="absolute inset-0 flex items-center justify-center bg-secondary/50 backdrop-blur-sm">
              <div className="text-center space-y-3">
                <Camera className="w-12 h-12 md:w-16 md:h-16 text-primary mx-auto animate-pulse" />
                <p className="text-base md:text-lg text-foreground font-medium">
                  Initializing camera...
                </p>
              </div>
            </div>
          )}
        </div>
      </main>

      {isActive && (
        <footer className="absolute bottom-0 left-0 right-0 z-20 p-4 md:p-6 bg-gradient-to-t from-background/90 to-transparent backdrop-blur-md">
          <div className="max-w-6xl mx-auto">
            <div className="bg-card/60 backdrop-blur-xl border border-border/50 rounded-2xl p-4 md:p-6 shadow-xl">
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
                {/* Camera Status */}
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary/20">
                    <Camera className="w-5 h-5 text-primary" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-muted-foreground">Camera</p>
                    <p className="text-sm font-semibold text-primary truncate">
                      Active
                    </p>
                  </div>
                </div>

                {/* Speech Status */}
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary/20">
                    <Mic className="w-5 h-5 text-primary" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-muted-foreground">Speech</p>
                    <p className="text-sm font-semibold text-primary truncate">
                      Enabled
                    </p>
                  </div>
                </div>

                {/* Speaking Status */}
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-10 h-10 rounded-full bg-accent/20">
                    <Waves
                      className={`w-5 h-5 text-accent ${
                        isSpeaking ? "animate-pulse" : ""
                      }`}
                    />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-muted-foreground">Status</p>
                    <p className="text-sm font-semibold text-accent truncate">
                      {isSpeaking ? "Speaking" : "Listening"}
                    </p>
                  </div>
                </div>

                {/* Detection Status */}
                <div className="flex items-center gap-3 col-span-2 lg:col-span-1">
                  <div className="flex items-center justify-center w-10 h-10 rounded-full bg-accent/20">
                    <Eye className="w-5 h-5 text-accent" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-muted-foreground">Detection</p>
                    <p className="text-sm font-semibold text-foreground truncate">
                      {detections[0] || "Scanning environment..."}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </footer>
      )}
    </div>
  );
}
