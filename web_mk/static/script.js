document.getElementById('uploadButton').addEventListener('click', async function() {
    const videoUpload = document.getElementById('videoUpload').files[0];
    if (!videoUpload) {
        alert("Please upload a video file first.");
        return;
    }

    const formData = new FormData();
    formData.append("video", videoUpload);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            document.getElementById('responseText').textContent = data.message;
            const audioElement = document.getElementById('audioPlayback');
            audioElement.src = data.audio_url;
            audioElement.load();
            audioElement.play();
        } else {
            const errorData = await response.json();
            alert(`Error: ${errorData.error}`);
        }
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred while processing the video.");
    }
});
