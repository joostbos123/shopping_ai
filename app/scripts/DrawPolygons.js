// Get references to the image and canvas elements
const image = document.getElementById('searchImage');
const canvas = document.getElementById('imagePolygonCanvas');
const ctx = canvas.getContext('2d');

// Set the canvas size to match the image size
canvas.width = image.width;
canvas.height = image.height;

// Define the polygon's vertices
const polygonVertices = [
    { x: 100, y: 100 },
    { x: 200, y: 100 },
    { x: 150, y: 200 }
];

// Function to draw the polygon on the canvas
function drawPolygon() {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear previous drawings
    ctx.drawImage(image, 0, 0); // Draw the image on the canvas

    ctx.beginPath();
    ctx.moveTo(polygonVertices[0].x, polygonVertices[0].y);

    for (let i = 1; i < polygonVertices.length; i++) {
        ctx.lineTo(polygonVertices[i].x, polygonVertices[i].y);
    }

    ctx.closePath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'red';
    ctx.stroke();
}

// Call the function to draw the polygon
drawPolygon();