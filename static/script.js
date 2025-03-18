const API_URL = "http://127.0.0.1:8000";
let isProcessing = false;
let controller = null;

document.getElementById("send-btn").addEventListener("click", handleSendButtonClick);
document.getElementById("attach-btn").addEventListener("click", () => document.getElementById("file-input").click());
document.getElementById("user-input").addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !isProcessing) {
        e.preventDefault();
        sendMessage();
    }
});
document.getElementById("file-input").addEventListener("change", handleFileUpload);

function updateSendButton(processing) {
    const sendBtn = document.getElementById("send-btn");
    const icon = sendBtn.querySelector("i");

    if (processing) {
        icon.className = "fas fa-stop";  // Ícono de detener en FontAwesome
    } else {
        icon.className = "fas fa-paper-plane";  // Ícono original de enviar
    }
}

function handleSendButtonClick() {
    if (isProcessing) {
        cancelProcessing();
    } else {
        sendMessage();
    }
}

async function sendMessage() {
    if (isProcessing) return;
    isProcessing = true;
    updateSendButton(true);

    const inputField = document.getElementById("user-input");
    const message = inputField.value.trim();
    if (!message) {
        isProcessing = false;
        updateSendButton(false);
        return;
    }

    addMessage("Tú", message, "user-message", "user-container", true);
    const thinkingMessage = addMessage("Chatbot IA", "Analizando...", "thinking-message", "ai-container");
    inputField.value = "";

    controller = new AbortController();

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message }),
            signal: controller.signal
        });

        if (!response.ok) throw new Error("Error en la respuesta del servidor");

        const data = await response.json();
        const cleanText = cleanResponse(data?.response);

        // 🔴 Se mantiene "Stop" hasta que termine de mostrar la respuesta
        await animateMessage(thinkingMessage, "Chatbot IA", cleanText || "Respuesta no disponible.");
        
    } catch (error) {
        if (error.name === "AbortError") {
            animateMessage(thinkingMessage, "Chatbot IA", "El proceso ha sido cancelado.");
        } else {
            animateMessage(thinkingMessage, "Chatbot IA", "Error al conectar con el servidor.");
            console.error("Error:", error);
        }
    }

    isProcessing = false; // 🔴 Solo aquí se vuelve false después de generar la respuesta
    updateSendButton(false); // Regresa a "Enviar"
}

async function handleFileUpload(event) {
    if (isProcessing) return;
    isProcessing = true;

    const file = event.target.files[0];
    if (!file) {
        isProcessing = false;
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    addMessage("Tú", `Archivo adjuntado: ${file.name}`, "user-message", "user-container", true);
    const processingMessage = addMessage("Chatbot IA", "Analizando...", "thinking-message", "ai-container");

    try {
        const response = await fetch(`${API_URL}/process-file/`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("Error en la respuesta del servidor");

        const blob = await response.blob();
        animateMessage(processingMessage, "Chatbot IA", "Descarga exitosa.");
        downloadSummary(blob, file.name);
    } catch (error) {
        animateMessage(processingMessage, "Chatbot IA", "Hubo un problema al analizar el documento.");
        console.error("Error:", error);
    }

    isProcessing = false;
    updateSendButton(false);
}

function downloadSummary(blob, fileName) {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${fileName}_resumen.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function cleanResponse(text) {
    return text || "Respuesta no disponible.";
}

function addMessage(sender, text, className, containerClass, isUser = false) {
    const chatBox = document.getElementById("chat-box");
    const messageContainer = document.createElement("div");
    messageContainer.classList.add("message-container", containerClass);
    if (isUser) messageContainer.classList.add("right");

    const messageElement = document.createElement("div");
    messageElement.classList.add("message", className);
    messageElement.innerHTML = `<strong>${sender}:</strong> ${text}`;

    messageContainer.appendChild(messageElement);
    chatBox.appendChild(messageContainer);
    chatBox.scrollTop = chatBox.scrollHeight;

    return messageElement;
}

// 🔴 Modifica animateMessage para permitir su interrupción
function animateMessage(element, sender, text) {
    return new Promise((resolve) => {
        element.innerHTML = `<strong>${sender}:</strong> `;
        let index = 0;

        // Guarda el intervalo en una variable global para cancelarlo
        window.typingInterval = setInterval(() => {
            if (index < text.length) {
                element.innerHTML += text[index];
                index++;
            } else {
                clearInterval(window.typingInterval);
                window.typingInterval = null;
                element.classList.remove("thinking-message");
                element.classList.add("ai-message");
                resolve();
            }
        }, 50);
    });
}

function cancelProcessing() {
    if (controller) {
        controller.abort();
        controller = null;
    }

    isProcessing = false;
    updateSendButton(false);

    // ⛔ Interrumpe la animación eliminando el intervalo activo
    if (window.typingInterval) {
        clearInterval(window.typingInterval);
        window.typingInterval = null;
    }

    // ⛔ Borra inmediatamente cualquier mensaje en progreso
    const thinkingMessages = document.querySelectorAll(".thinking-message, .ai-message");
    thinkingMessages.forEach(msg => {
        msg.innerHTML = `<strong>Chatbot IA:</strong> El proceso ha sido cancelado.`;
        msg.classList.remove("thinking-message");
        msg.classList.add("ai-message"); // Asegura que se vea como mensaje finalizado
    });
}
